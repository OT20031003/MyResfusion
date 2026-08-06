"""WR.tex の推論と、AWGNあり・なしのADJSCC baselineを評価する。"""

import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from adjscc_signal_data import list_images, load_crop
from adjscc_signal_data_wr import MINIMUM_CHANNEL_SNR_DB, sigma_from_snr_db
from model import WRADJSCCSignalResfusion
from model.denoising_module import RDDM_Unet
from test_adjscc_signal_resfusion import calculate_perceptual_metrics, psnr
from variance_scheduler import CosineProScheduler, LinearProScheduler

def load_model(path, device):
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu")
    hp = checkpoint["hyper_parameters"]
    if hp.get("wr_formula_version") != "WR.tex_v1":
        raise ValueError("WR.tex方式で学習したcheckpointではありません")
    channels = int(hp["transmit_channel_num"])
    denoiser = RDDM_Unet(dim=int(hp["dim"]), out_dim=channels, channels=channels,
        input_condition=True, input_condition_channels=channels,
        resnet_block_groups=int(hp["resnet_block_groups"]), channel_snr_condition=True,
        channel_snr_min_db=float(hp["channel_snr_min_db"]),
        channel_snr_max_db=float(hp["channel_snr_max_db"]))
    scheduler = LinearProScheduler(int(hp["T"])) if hp["noise_schedule"] == "LinearPro" else CosineProScheduler(int(hp["T"]))
    model = WRADJSCCSignalResfusion(denoiser, scheduler, **hp)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    for name in ("alphas_hat", "alphas", "betas", "betas_hat", "alphas_hat_t_minus_1"):
        setattr(model, name, getattr(model, name).to(device))
    return model, hp


def complex_mean_power(signal):
    """隣り合う2実数成分を1複素symbolとした平均電力。"""
    flattened = np.asarray(signal, dtype=np.float32).reshape(len(signal), -1)
    if flattened.shape[1] % 2:
        raise ValueError("latent element count must be even")
    half = flattened.shape[1] // 2
    symbols = flattened[:, :half] + 1j * flattened[:, half:]
    return float(np.mean(np.abs(symbols) ** 2))

def main(args):
    if args.channel_snr_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError(f"--channel_snr_dbは{MINIMUM_CHANNEL_SNR_DB:.6f} dB以上にしてください")
    if args.metrics_batch_size < 1:
        raise ValueError("--metrics_batch_sizeは1以上にしてください")
    sampling_mode = "deterministic" if args.deterministic else args.sampling_mode
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, hp = load_model(args.resfusion_ckpt, device)
    from ADJSCC.adjscc_module import ADJSCCCodec
    codec = ADJSCCCodec(args.adjscc_weights or hp["adjscc_weights"],
        int(hp["transmit_channel_num"]), int(hp.get("image_size", 256)),
        seed=args.seed, device=args.tf_device)
    paths = list_images(args.input_dir)
    if args.limit is not None: paths = paths[:args.limit]
    if len(paths) < 2:
        raise ValueError("FID計算には2枚以上の評価画像が必要です")
    output = Path(args.output_dir).expanduser(); output.mkdir(parents=True, exist_ok=True)
    route_names = (
        "awgn_resfusion_wr", "awgn_no_resfusion_low",
        "awgn_no_resfusion_high", "high2high_no_resfusion",
        "actual_channel_adjscc",
    )
    route_dirs = {name: output / name for name in route_names}
    for directory in route_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    crop_rng, noise_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1)
    sigma = float(sigma_from_snr_db(torch.tensor(args.channel_snr_db)))
    low_snr = float(args.low_snr if args.low_snr is not None else hp["low_snr"])
    high_snr = float(args.high_snr if args.high_snr is not None else hp["high_snr"])
    rows, powers = [], {name: [] for name in route_names}
    for index, path in enumerate(paths):
        image = load_crop(path, codec.image_size, False, crop_rng)
        z_low = codec.encode(image[None], low_snr)
        # WR.tex の n~N(0,1) を全実数成分に適用。受信後は正規化・clipしない。
        channel_noise = noise_rng.standard_normal(z_low.shape).astype(np.float32)
        y = z_low + sigma * channel_noise
        y_tensor = torch.from_numpy(y.transpose(0, 3, 1, 2)).to(device)
        with torch.inference_mode():
            predicted = model.infer_from_received(y_tensor, torch.tensor([sigma], device=device),
                torch.tensor([args.channel_snr_db], device=device),
                stochastic=sampling_mode == "stochastic")
        latent = predicted.cpu().numpy().transpose(0, 2, 3, 1)
        if args.output_power_normalize: latent = codec.power_normalize(latent)
        reconstructed = codec.decode(latent, high_snr)[0]

        # Resfusionを通さないbaselineは、提案法と同じ受信信号を
        # low/high SNR decoderにそのまま入れる。
        direct_low = codec.decode(y, low_snr)[0]
        direct_high = codec.decode(y, high_snr)[0]

        # high-SNR条件でencodeした信号を、AWGNとResfusionを通さず
        # high-SNR decoderへ直接入れる。
        high2high_latent = codec.encode(image[None], high_snr)
        high2high = codec.decode(high2high_latent, high_snr)[0]

        # 実チャネルSNR用ADJSCC baseline。公平な比較のため同じ
        # 実数Gaussian標本を使い、WRのチャネル定義を変えない。
        actual_tx = codec.encode(image[None], args.channel_snr_db)
        actual_received = actual_tx + sigma * channel_noise
        actual = codec.decode(actual_received, args.channel_snr_db)[0]
        outputs = {
            "awgn_resfusion_wr": (reconstructed, latent),
            "awgn_no_resfusion_low": (direct_low, y),
            "awgn_no_resfusion_high": (direct_high, y),
            "high2high_no_resfusion": (high2high, high2high_latent),
            "actual_channel_adjscc": (actual, actual_received),
        }
        row = [path.name]
        for route_name in route_names:
            decoded, route_latent = outputs[route_name]
            destination = route_dirs[route_name] / f"{path.stem}_reconstructed.png"
            Image.fromarray(np.rint(decoded).astype(np.uint8)).save(destination)
            score = psnr(image, decoded)
            power = complex_mean_power(route_latent)
            powers[route_name].append(power)
            row.extend((str(destination.relative_to(output)), score, power))
        rows.append(row)
        print(
            f"[{index+1}/{len(paths)}] {path.name}: "
            f"WR={row[2]:.3f}, low={row[5]:.3f}, "
            f"high={row[8]:.3f}, high2high={row[11]:.3f}, "
            f"actual={row[14]:.3f} dB"
        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["input"]
        for route_name in route_names:
            header.extend((f"{route_name}_image", f"{route_name}_psnr_db", f"{route_name}_latent_power"))
        writer.writerow(header); writer.writerows(rows)

    metrics_device = torch.device(args.metrics_device)
    summary = []
    for route_index, route_name in enumerate(route_names):
        scores = [float(row[2 + route_index * 3]) for row in rows]
        perceptual = calculate_perceptual_metrics(
            paths, route_dirs[route_name], codec.image_size, metrics_device,
            args.seed, args.metrics_batch_size,
        )
        summary.append((
            route_name, args.channel_snr_db, sampling_mode,
            args.output_power_normalize, float(np.mean(powers[route_name])),
            float(np.mean(scores)), perceptual["lpips"], perceptual["dists"],
            perceptual["fid"],
        ))
        print(
            f"{route_name}: PSNR={np.mean(scores):.3f}, "
            f"LPIPS={perceptual['lpips']:.6f}, DISTS={perceptual['dists']:.6f}, "
            f"FID={perceptual['fid']:.3f}, power={np.mean(powers[route_name]):.6f}"
        )
    with (output / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "route", "channel_snr_db", "sampling_mode", "output_power_normalize",
            "mean_latent_power", "mean_psnr_db", "mean_lpips", "mean_dists", "fid",
        ))
        writer.writerows(summary)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resfusion_ckpt", required=True); p.add_argument("--channel_snr_db", required=True, type=float)
    p.add_argument("--adjscc_weights"); p.add_argument("--input_dir", default="../datasets/Raindrop/test_a/gt")
    p.add_argument("--output_dir", default="wr_signal_inference"); p.add_argument("--device"); p.add_argument("--tf_device")
    p.add_argument("--low_snr", type=float); p.add_argument("--high_snr", type=float)
    p.add_argument("--metrics_device", default="cpu")
    p.add_argument("--metrics_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2024); p.add_argument("--limit", type=int)
    p.add_argument("--deterministic", action="store_true",
                   help="各逆stepのposterior noiseを0にする")
    p.add_argument("--sampling-mode", choices=("stochastic", "deterministic"), default="stochastic",
                   help="--deterministicの明示形。両方指定時はdeterministicを優先")
    p.add_argument("--output-power-normalize", dest="output_power_normalize", action="store_true")
    p.add_argument("--no-output-power-normalize", dest="output_power_normalize", action="store_false")
    p.set_defaults(output_power_normalize=True); return p

if __name__ == "__main__": main(parser().parse_args())
