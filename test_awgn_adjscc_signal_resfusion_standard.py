"""received-y checkpointを標準Resfusion初期化で評価する新規entrypoint。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adjscc_signal_data import list_images, load_crop
from adjscc_signal_data_standard import from_model_domain, to_model_domain
from test_adjscc_signal_resfusion import calculate_perceptual_metrics, load_resfusion, psnr


P = 1.0
MINIMUM_CHANNEL_SNR_DB = -4.0
EXPECTED_METADATA = {
    "degraded_input_mode": "received_y",
    "residual_definition": "y_minus_s_high",
    "channel_noise_in_diffusion": False,
    "standard_resfusion_initialization": True,
    "signal_processing_version": "awgn_received_y_standard_resfusion_v1",
}


def real_to_complex(signals: np.ndarray) -> np.ndarray:
    flattened = np.asarray(signals, dtype=np.float32).reshape(len(signals), -1)
    if flattened.shape[1] % 2:
        raise ValueError("latent element count must be even")
    half = flattened.shape[1] // 2
    return flattened[:, :half] + 1j * flattened[:, half:]


def complex_to_real(symbols: np.ndarray, shape) -> np.ndarray:
    return np.concatenate([symbols.real, symbols.imag], axis=1).reshape(shape).astype(np.float32)


def complex_standard_normal(rng: np.random.Generator, shape) -> np.ndarray:
    scale = 1.0 / np.sqrt(2.0)
    return (rng.normal(0.0, scale, shape) + 1j * rng.normal(0.0, scale, shape)).astype(np.complex64)


def add_channel_awgn(
    transmitted: np.ndarray, channel_snr_db: float, rng: np.random.Generator
):
    if channel_snr_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError("channel_snr_db must be at least -4 dB")
    symbols = real_to_complex(transmitted)
    channel_epsilon = complex_standard_normal(rng, symbols.shape)
    channel_std = np.sqrt(P * 10.0 ** (-channel_snr_db / 10.0))
    channel_noise = channel_std * channel_epsilon
    received = symbols + channel_noise
    # receivedへpower normalizationやclampは適用しない。
    return complex_to_real(received, transmitted.shape), channel_epsilon, channel_std


def standard_resfusion_initialization(
    received: np.ndarray, start_alpha_bar: float, rng: np.random.Generator
):
    """channel noiseと無関係な標準GaussianでResfusion開始状態を作る。"""
    if not 0.0 < start_alpha_bar < 1.0:
        raise ValueError("start_alpha_bar must be in (0, 1)")
    received_symbols = real_to_complex(received)
    diffusion_epsilon = complex_standard_normal(rng, received_symbols.shape)
    initial_symbols = (
        np.sqrt(start_alpha_bar) * received_symbols
        + np.sqrt(1.0 - start_alpha_bar) * diffusion_epsilon
    )
    return complex_to_real(initial_symbols, received.shape), diffusion_epsilon


def inverse_resfusion(
    model, initial_state: torch.Tensor, received_condition: torch.Tensor,
    channel_snr_db: float, sampling_mode: str,
) -> torch.Tensor:
    state = initial_state
    for t in range(model.T_acc - 1, -1, -1):
        time = torch.full((state.shape[0],), t, dtype=torch.long, device=state.device)
        snr = torch.full(
            (state.shape[0],), channel_snr_db,
            dtype=state.dtype, device=state.device,
        )
        predicted_resnoise = model.denoising_module(
            x=state, time=time, input_cond=received_condition,
            channel_snr_db=snr,
        )
        if sampling_mode == "deterministic" or t == 0:
            posterior_noise = torch.zeros_like(state)
        else:
            posterior_noise = torch.randn_like(state)
        state = (
            (state - model.betas[t] / torch.sqrt(1.0 - model.alphas_hat[t])
             * predicted_resnoise) / torch.sqrt(model.alphas[t])
            + torch.sqrt(model.betas_hat[t]) * posterior_noise
        )
    return state


def complex_mean_power(signal: np.ndarray) -> float:
    return float(np.mean(np.abs(real_to_complex(signal)) ** 2))


def validate_checkpoint_metadata(settings: dict) -> None:
    mismatches = {
        key: (settings.get(key), expected)
        for key, expected in EXPECTED_METADATA.items()
        if settings.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "checkpoint is not trained for standard received-y Resfusion; retraining is required: "
            + repr(mismatches)
        )


def main(args) -> None:
    if args.channel_snr_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError("--channel_snr_db must be at least -4 dB")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, settings = load_resfusion(args.resfusion_ckpt, device)
    validate_checkpoint_metadata(settings)
    start_alpha_bar = float(model.alphas_hat[model.T_acc - 1].detach().cpu())
    low_snr = float(args.low_snr if args.low_snr is not None else settings["low_snr"])
    high_snr = float(args.high_snr if args.high_snr is not None else settings["high_snr"])
    weights = args.adjscc_weights or settings["adjscc_weights"]
    latent_min = float(settings["latent_min"])
    latent_max = float(settings["latent_max"])
    from ADJSCC.adjscc_module import ADJSCCCodec
    codec = ADJSCCCodec(
        weights, int(settings["transmit_channel_num"]),
        int(settings.get("image_size", 256)), seed=args.seed, device=args.tf_device,
    )
    paths = list_images(args.input_dir)
    if args.limit is not None:
        paths = paths[:args.limit]
    if len(paths) < 2:
        raise ValueError("at least two images are required for FID")
    output_root = Path(args.output_dir).expanduser()
    route_names = (
        "awgn_resfusion_standard", "awgn_no_resfusion_low",
        "awgn_no_resfusion_high", "actual_channel_adjscc",
    )
    route_dirs = {name: output_root / name for name in route_names}
    for directory in route_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    crop_rng = np.random.default_rng(args.seed)
    channel_rng = np.random.default_rng(args.seed + 1)
    diffusion_rng = np.random.default_rng(args.seed + 2)
    rows, powers = [], {name: [] for name in route_names}
    channel_std = 0.0
    for index, path in enumerate(paths):
        image = load_crop(path, codec.image_size, False, crop_rng)
        transmitted = codec.encode(image[None], low_snr)
        received, channel_epsilon, channel_std = add_channel_awgn(
            transmitted, args.channel_snr_db, channel_rng
        )
        received_model = to_model_domain(
            torch.from_numpy(np.transpose(received, (0, 3, 1, 2))),
            latent_min, latent_max,
        )
        # 学習時と同じmodel-domainで標準初期化する。固定affineのoffsetを
        # diffusion stateへ二重に混ぜない。
        initial_model_numpy, diffusion_epsilon = standard_resfusion_initialization(
            received_model.numpy(), start_alpha_bar, diffusion_rng
        )
        # 明示的に別々に生成した標本であり、memory共有もしない。
        if np.shares_memory(channel_epsilon, diffusion_epsilon):
            raise RuntimeError("channel and diffusion noise unexpectedly share memory")
        received_model = received_model.to(device)
        initial_model = torch.from_numpy(initial_model_numpy).to(device)
        with torch.inference_mode():
            predicted_model = inverse_resfusion(
                model, initial_model, received_model,
                args.channel_snr_db, args.sampling_mode,
            )
        predicted_raw = from_model_domain(
            predicted_model.cpu(), latent_min, latent_max
        ).permute(0, 2, 3, 1).numpy()
        if args.output_power_normalize:
            predicted_raw = codec.power_normalize(predicted_raw)
        proposed = codec.decode(predicted_raw, high_snr)[0]
        direct_low = codec.decode(received, low_snr)[0]
        direct_high = codec.decode(received, high_snr)[0]

        actual_tx = codec.encode(image[None], args.channel_snr_db)
        actual_symbols = real_to_complex(actual_tx)
        actual_received = complex_to_real(
            actual_symbols + channel_std * channel_epsilon, actual_tx.shape
        )
        actual = codec.decode(actual_received, args.channel_snr_db)[0]
        outputs = {
            "awgn_resfusion_standard": (proposed, predicted_raw),
            "awgn_no_resfusion_low": (direct_low, received),
            "awgn_no_resfusion_high": (direct_high, received),
            "actual_channel_adjscc": (actual, actual_received),
        }
        row = [path.name]
        for route_name in route_names:
            decoded, latent = outputs[route_name]
            destination = route_dirs[route_name] / f"{path.stem}_reconstructed.png"
            Image.fromarray(np.rint(decoded).astype(np.uint8)).save(destination)
            score = psnr(image, decoded)
            power = complex_mean_power(latent)
            powers[route_name].append(power)
            row.extend((str(destination.relative_to(output_root)), score, power))
        rows.append(row)
        print(f"[{index + 1}/{len(paths)}] {path.name}: standard={row[2]:.3f} dB")

    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["input"]
        for name in route_names:
            header.extend((f"{name}_image", f"{name}_psnr_db", f"{name}_latent_power"))
        writer.writerow(header)
        writer.writerows(rows)

    summary = []
    metrics_device = torch.device(args.metrics_device)
    for route_index, route_name in enumerate(route_names):
        scores = [float(row[2 + route_index * 3]) for row in rows]
        perceptual = calculate_perceptual_metrics(
            paths, route_dirs[route_name], codec.image_size, metrics_device,
            args.seed, args.metrics_batch_size,
        )
        summary.append((
            route_name, args.channel_snr_db, start_alpha_bar,
            args.sampling_mode, args.output_power_normalize,
            float(np.mean(powers[route_name])), float(np.mean(scores)),
            perceptual["lpips"], perceptual["dists"], perceptual["fid"],
        ))
        print(
            f"{route_name}: PSNR={np.mean(scores):.3f}, "
            f"LPIPS={perceptual['lpips']:.6f}, DISTS={perceptual['dists']:.6f}, "
            f"FID={perceptual['fid']:.3f}, power={np.mean(powers[route_name]):.6f}"
        )
    with (output_root / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "route", "channel_snr_db", "start_alpha_bar", "sampling_mode",
            "output_power_normalize", "mean_latent_power", "mean_psnr_db",
            "mean_lpips", "mean_dists", "fid",
        ))
        writer.writerows(summary)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--resfusion_ckpt", required=True)
    value.add_argument("--channel_snr_db", required=True, type=float)
    value.add_argument("--adjscc_weights", default=None)
    value.add_argument("--input_dir", default="../datasets/Raindrop/test_a/gt")
    value.add_argument("--output_dir", default="standard_received_y_eval")
    value.add_argument("--low_snr", type=float, default=None)
    value.add_argument("--high_snr", type=float, default=None)
    value.add_argument("--device", default=None)
    value.add_argument("--tf_device", default=None)
    value.add_argument("--metrics_device", default="cpu")
    value.add_argument("--metrics_batch_size", type=int, default=8)
    value.add_argument("--seed", type=int, default=2024)
    value.add_argument("--limit", type=int, default=None)
    value.add_argument(
        "--sampling-mode", choices=("stochastic", "deterministic"),
        default="stochastic",
    )
    value.add_argument("--output-power-normalize", dest="output_power_normalize", action="store_true")
    value.add_argument("--no-output-power-normalize", dest="output_power_normalize", action="store_false")
    value.set_defaults(output_power_normalize=True)
    return value


if __name__ == "__main__":
    main(parser().parse_args())
