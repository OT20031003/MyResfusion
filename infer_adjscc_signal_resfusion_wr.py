"""WR.tex の受信信号初期化と逆過程を実行する推論 entrypoint。"""

import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from adjscc_signal_data import list_images, load_crop
from adjscc_signal_data_wr import MINIMUM_CHANNEL_SNR_DB, sigma_from_snr_db
from model import WRADJSCCSignalResfusion
from model.denoising_module import RDDM_Unet
from test_adjscc_signal_resfusion import psnr
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

def main(args):
    if args.channel_snr_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError(f"--channel_snr_dbは{MINIMUM_CHANNEL_SNR_DB:.6f} dB以上にしてください")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, hp = load_model(args.resfusion_ckpt, device)
    from ADJSCC.adjscc_module import ADJSCCCodec
    codec = ADJSCCCodec(args.adjscc_weights or hp["adjscc_weights"],
        int(hp["transmit_channel_num"]), int(hp.get("image_size", 256)),
        seed=args.seed, device=args.tf_device)
    paths = list_images(args.input_dir)
    if args.limit is not None: paths = paths[:args.limit]
    output = Path(args.output_dir).expanduser(); output.mkdir(parents=True, exist_ok=True)
    crop_rng, noise_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1)
    sigma = float(sigma_from_snr_db(torch.tensor(args.channel_snr_db)))
    rows = []
    for index, path in enumerate(paths):
        image = load_crop(path, codec.image_size, False, crop_rng)
        z_low = codec.encode(image[None], float(hp["low_snr"]))
        # WR.tex の n~N(0,1) を全実数成分に適用。受信後は正規化・clipしない。
        y = z_low + sigma * noise_rng.standard_normal(z_low.shape).astype(np.float32)
        y_tensor = torch.from_numpy(y.transpose(0, 3, 1, 2)).to(device)
        with torch.inference_mode():
            predicted = model.infer_from_received(y_tensor, torch.tensor([sigma], device=device),
                torch.tensor([args.channel_snr_db], device=device), stochastic=not args.deterministic)
        latent = predicted.cpu().numpy().transpose(0, 2, 3, 1)
        if args.output_power_normalize: latent = codec.power_normalize(latent)
        reconstructed = codec.decode(latent, float(hp["high_snr"]))[0]
        destination = output / f"{path.stem}_reconstructed.png"
        Image.fromarray(np.rint(reconstructed).astype(np.uint8)).save(destination)
        score = psnr(image, reconstructed); rows.append((path.name, destination.name, score))
        print(f"[{index+1}/{len(paths)}] {path.name}: PSNR={score:.3f} dB")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(("input", "output", "psnr_db")); writer.writerows(rows)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resfusion_ckpt", required=True); p.add_argument("--channel_snr_db", required=True, type=float)
    p.add_argument("--adjscc_weights"); p.add_argument("--input_dir", default="../datasets/Raindrop/test_a/gt")
    p.add_argument("--output_dir", default="wr_signal_inference"); p.add_argument("--device"); p.add_argument("--tf_device")
    p.add_argument("--seed", type=int, default=2024); p.add_argument("--limit", type=int)
    p.add_argument("--deterministic", action="store_true",
                   help="各逆stepのposterior noiseを0にする")
    p.add_argument("--output-power-normalize", dest="output_power_normalize", action="store_true")
    p.add_argument("--no-output-power-normalize", dest="output_power_normalize", action="store_false")
    p.set_defaults(output_power_normalize=True); return p

if __name__ == "__main__": main(parser().parse_args())
