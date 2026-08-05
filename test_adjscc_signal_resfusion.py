"""ADJSCC Encoder(low SNR) -> Resfusion -> Decoder(high SNR)をテストする。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adjscc_signal_data import list_images, load_crop
from model import LatentResfusion
from model.denoising_module import RDDM_Unet
from variance_scheduler import CosineProScheduler, LinearProScheduler


def load_resfusion(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    required = ("transmit_channel_num", "T", "noise_schedule", "latent_min", "latent_max")
    missing = [key for key in required if key not in hparams]
    if missing:
        raise RuntimeError(f"MyResfusion checkpointに必要な設定がありません: {missing}")
    channels = int(hparams["transmit_channel_num"])
    denoiser = RDDM_Unet(
        dim=int(hparams.get("dim", 64)), out_dim=channels, channels=channels,
        input_condition=True, input_condition_channels=channels,
        resnet_block_groups=int(hparams.get("resnet_block_groups", 8)),
    )
    scheduler_type = hparams["noise_schedule"]
    scheduler = LinearProScheduler(int(hparams["T"])) if scheduler_type == "LinearPro" else CosineProScheduler(int(hparams["T"]))
    model = LatentResfusion(denoising_module=denoiser, variance_scheduler=scheduler, **hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    for name in ("alphas_hat", "alphas", "betas", "betas_hat", "alphas_hat_t_minus_1"):
        setattr(model, name, getattr(model, name).to(device))
    return model, hparams


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main(args) -> None:
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, settings = load_resfusion(args.resfusion_ckpt, device)
    low_snr = float(args.low_snr if args.low_snr is not None else settings["low_snr"])
    high_snr = float(args.high_snr if args.high_snr is not None else settings["high_snr"])
    adjscc_weights = args.adjscc_weights or settings["adjscc_weights"]
    channels = int(settings["transmit_channel_num"])
    image_size = int(settings.get("image_size", 256))
    latent_min, latent_max = float(settings["latent_min"]), float(settings["latent_max"])
    latent_scale = latent_max - latent_min

    # TensorFlowはここで初めて読み込み、ADJSCC自体は常に凍結・チャネルなしで使う。
    from ADJSCC.adjscc_module import ADJSCCCodec
    codec = ADJSCCCodec(adjscc_weights, channels, image_size, seed=args.seed or 2024, device=args.tf_device)

    image_dir = Path(args.input_dir).expanduser()
    paths = list_images(str(image_dir))
    if args.limit is not None:
        paths = paths[: args.limit]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    resfusion_dir = output_dir / "resfusion_high_decoder"
    direct_low_dir = output_dir / "no_resfusion_low_decoder"
    direct_high_dir = output_dir / "no_resfusion_high_decoder"
    for directory in (resfusion_dir, direct_low_dir, direct_high_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed or 2024)
    rows = []
    for index, path in enumerate(paths):
        # 学習時と同じ256x256。テストでは再現可能な中央cropとする。
        image = load_crop(path, image_size, random_crop=False, rng=rng)
        low_signal = codec.encode(image[None], low_snr)

        # 比較用：Resfusionを通さず、同じlow-SNR latentを2種類の
        # Decoder条件へ直接入力する。low_signalはすでに電力正規化済み。
        direct_low = codec.decode(low_signal, low_snr)[0]
        direct_high = codec.decode(low_signal, high_snr)[0]

        low_nchw = torch.from_numpy(np.transpose(low_signal, (0, 3, 1, 2))).float().to(device)
        low_normalized = ((low_nchw - latent_min) / latent_scale).clamp(0.0, 1.0)
        with torch.inference_mode():
            predicted = model.generate(low_normalized * 2.0 - 1.0)
        predicted01 = ((predicted + 1.0) / 2.0).clamp(0.0, 1.0)
        predicted_raw = predicted01 * latent_scale + latent_min
        predicted_nhwc = predicted_raw.permute(0, 2, 3, 1).cpu().numpy()
        # Decoderが学習時に受け取っていた平均電力1の信号へ戻す。AWGNは加えない。
        predicted_nhwc = codec.power_normalize(predicted_nhwc)
        reconstructed = codec.decode(predicted_nhwc, high_snr)[0]

        filename = f"{path.stem}_reconstructed.png"
        resfusion_path = resfusion_dir / filename
        direct_low_path = direct_low_dir / filename
        direct_high_path = direct_high_dir / filename
        Image.fromarray(np.rint(reconstructed).astype(np.uint8)).save(resfusion_path)
        Image.fromarray(np.rint(direct_low).astype(np.uint8)).save(direct_low_path)
        Image.fromarray(np.rint(direct_high).astype(np.uint8)).save(direct_high_path)

        resfusion_score = psnr(image, reconstructed)
        direct_low_score = psnr(image, direct_low)
        direct_high_score = psnr(image, direct_high)
        rows.append((
            path.name,
            str(resfusion_path.relative_to(output_dir)), resfusion_score,
            str(direct_low_path.relative_to(output_dir)), direct_low_score,
            str(direct_high_path.relative_to(output_dir)), direct_high_score,
        ))
        print(
            f"[{index + 1}/{len(paths)}] {path.name}: "
            f"Resfusion->high={resfusion_score:.3f} dB, "
            f"low->low={direct_low_score:.3f} dB, "
            f"low->high={direct_high_score:.3f} dB"
        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "input",
            "resfusion_high_image", "resfusion_high_psnr_db",
            "no_resfusion_low_image", "no_resfusion_low_psnr_db",
            "no_resfusion_high_image", "no_resfusion_high_psnr_db",
        ))
        writer.writerows(rows)
    print(f"Mean Resfusion->high PSNR: {np.mean([row[2] for row in rows]):.3f} dB")
    print(f"Mean low->low PSNR: {np.mean([row[4] for row in rows]):.3f} dB")
    print(f"Mean low->high PSNR: {np.mean([row[6] for row in rows]):.3f} dB")
    print(f"Saved: {output_dir.resolve()}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--resfusion_ckpt", required=True)
    value.add_argument("--adjscc_weights", default=None, help="省略時はResfusion checkpoint内のパスを使う")
    value.add_argument("--input_dir", default="../datasets/Raindrop/test_a/gt")
    value.add_argument("--output_dir", default="my_resfusion_eval")
    value.add_argument("--low_snr", type=float, default=None)
    value.add_argument("--high_snr", type=float, default=None)
    value.add_argument("--device", default=None, help="PyTorch device。例: cuda:0, cpu")
    value.add_argument("--tf_device", default=None, help="TensorFlow device。例: /GPU:0, /CPU:0")
    value.add_argument("--seed", type=int, default=2024)
    value.add_argument("--limit", type=int, default=None)
    return value


if __name__ == "__main__":
    main(parser().parse_args())
