"""物理AWGNをResfusion開始noiseの一部として利用する潜在信号テスト。

注意: このnoise-completion初期化は、学習時に独立生成するchannel AWGNと
diffusion noiseの分布とは完全には同一でない。既存評価方式を維持している。
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adjscc_signal_data import list_images, load_crop
from test_adjscc_signal_resfusion import (
    calculate_perceptual_metrics,
    load_resfusion,
    psnr,
)


P = 1.0
MINIMUM_INPUT_SNR_DB = -4.0


def real_to_complex(signals: np.ndarray) -> np.ndarray:
    """ADJSCC Channelと同じflatten順で実数latentを複素シンボル化する。"""
    flattened = np.asarray(signals, dtype=np.float32).reshape(len(signals), -1)
    if flattened.shape[1] % 2:
        raise ValueError("潜在信号の要素数は偶数である必要があります")
    half = flattened.shape[1] // 2
    return flattened[:, :half] + 1j * flattened[:, half:]


def complex_to_real(symbols: np.ndarray, shape) -> np.ndarray:
    flattened = np.concatenate([symbols.real, symbols.imag], axis=1)
    return flattened.reshape(shape).astype(np.float32)


def complex_standard_normal(rng: np.random.Generator, shape) -> np.ndarray:
    """E[|epsilon|^2]=1の円対称複素Gaussian noiseを生成する。"""
    scale = 1.0 / np.sqrt(2.0)
    return (
        rng.normal(0.0, scale, shape) + 1j * rng.normal(0.0, scale, shape)
    ).astype(np.complex64)


def channel_noise_aware_initialization(
    transmitted: np.ndarray,
    channel_snr_db: float,
    rng: np.random.Generator,
):
    """AWGN受信信号yとnoise completionによるu_T'を構成する。

    P=1、alpha_bar_T'=1/4。AWGN後からu_T'までclip・再正規化しない。
    """
    if channel_snr_db < MINIMUM_INPUT_SNR_DB:
        raise ValueError(
            f"--channel_snr_db={channel_snr_db:g} dBは未定義です。"
            f" {MINIMUM_INPUT_SNR_DB:g} dB以上を指定してください。"
        )
    symbols = real_to_complex(transmitted)
    rho_inverse = 10.0 ** (-channel_snr_db / 10.0)
    channel_std = np.sqrt(P * rho_inverse)
    channel_epsilon = complex_standard_normal(rng, symbols.shape)
    channel_noise = channel_std * channel_epsilon
    received = symbols + channel_noise

    additional_variance = P * (3.0 - rho_inverse) / 4.0
    if additional_variance < 0.0:
        raise ValueError(
            "追加noise分散が負になりました。channel SNRまたは式の設定を確認してください。"
        )
    additional_std = np.sqrt(additional_variance)
    additional_noise = additional_std * complex_standard_normal(rng, symbols.shape)
    initial = 0.5 * received + additional_noise
    return (
        complex_to_real(received, transmitted.shape),
        complex_to_real(initial, transmitted.shape),
        channel_std,
        additional_std,
        channel_epsilon,
    )


def inverse_resfusion_from_initial(
    model, initial_state: torch.Tensor, received_condition: torch.Tensor,
    channel_snr_db: float,
) -> torch.Tensor:
    """指定したu_T'とyからepsilon予測Resfusionの逆拡散だけを実行する。"""
    if model.mode != "epsilon":
        raise ValueError("このテストは--mode epsilonで学習したcheckpointだけを対象とします")
    state = initial_state
    for t in range(model.T_acc - 1, -1, -1):
        alpha_t = model.alphas[t]
        alpha_hat_t = model.alphas_hat[t]
        beta_t = model.betas[t]
        beta_hat_t = model.betas_hat[t]
        time = torch.full(
            (state.shape[0],), t, dtype=torch.long, device=state.device
        )
        predicted_resnoise = model.denoising_module(
            x=state, time=time, input_cond=received_condition,
            channel_snr_db=torch.full(
                (state.shape[0],), channel_snr_db,
                device=state.device, dtype=state.dtype,
            ),
        )
        noise = torch.zeros_like(state) if t == 0 else torch.randn_like(state)
        state = (
            (state - beta_t / torch.sqrt(1.0 - alpha_hat_t) * predicted_resnoise)
            / torch.sqrt(alpha_t)
            + torch.sqrt(beta_hat_t) * noise
        )
    return state


def to_model_domain(signal: np.ndarray, latent_min: float, latent_scale: float, device):
    """raw NHWC latentをcheckpoint学習時のNCHW座標へ写像する。clipしない。"""
    value = torch.from_numpy(np.transpose(signal, (0, 3, 1, 2))).float().to(device)
    return 2.0 * (value - latent_min) / latent_scale - 1.0


def main(args) -> None:
    if args.channel_snr_db < MINIMUM_INPUT_SNR_DB:
        raise ValueError(
            f"--channel_snr_dbは{MINIMUM_INPUT_SNR_DB:g} dB以上にしてください"
        )
    if args.metrics_batch_size < 1:
        raise ValueError("--metrics_batch_sizeは1以上にしてください")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, settings = load_resfusion(args.resfusion_ckpt, device)
    if model.T_acc < 1:
        raise RuntimeError("checkpointのResfusion開始stepが不正です")
    start_alpha_bar = float(model.alphas_hat[model.T_acc - 1].detach().cpu())
    if not np.isclose(start_alpha_bar, 0.25, atol=1e-4):
        raise ValueError(
            "このnoise completionはalpha_bar_T'=1/4専用ですが、"
            f" checkpointの開始値は{start_alpha_bar:.8f}です"
        )
    low_snr = float(args.low_snr if args.low_snr is not None else settings["low_snr"])
    high_snr = float(args.high_snr if args.high_snr is not None else settings["high_snr"])
    weights = args.adjscc_weights or settings["adjscc_weights"]
    channels = int(settings["transmit_channel_num"])
    image_size = int(settings.get("image_size", 256))
    latent_min = float(settings["latent_min"])
    latent_scale = float(settings["latent_max"]) - latent_min

    from ADJSCC.adjscc_module import ADJSCCCodec

    codec = ADJSCCCodec(
        weights, channels, image_size, seed=args.seed, device=args.tf_device
    )
    paths = list_images(args.input_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    if len(paths) < 2:
        raise ValueError("FID計算には2枚以上の画像が必要です")
    output_dir = Path(args.output_dir).expanduser()
    reconstructed_dir = output_dir / "awgn_noise_completion"
    direct_low_dir = output_dir / "awgn_no_resfusion_low_decoder"
    direct_high_dir = output_dir / "awgn_no_resfusion_high_decoder"
    actual_channel_dir = output_dir / "actual_channel_adjscc"
    for directory in (
        reconstructed_dir, direct_low_dir, direct_high_dir, actual_channel_dir
    ):
        directory.mkdir(parents=True, exist_ok=True)

    crop_rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1)
    rows = []
    channel_std = additional_std = 0.0
    for index, path in enumerate(paths):
        image = load_crop(path, image_size, random_crop=False, rng=crop_rng)
        transmitted = codec.encode(image[None], low_snr)
        received, initial, channel_std, additional_std, channel_epsilon = (
            channel_noise_aware_initialization(
                transmitted, args.channel_snr_db, noise_rng
            )
        )

        # Resfusionなしの2経路にも提案法と同じAWGN受信信号yを入力する。
        # AWGN後のreceivedには再電力正規化・clampを適用しない。
        direct_low = codec.decode(received, low_snr)[0]
        direct_high = codec.decode(received, high_snr)[0]

        # Actual-channel baseline:
        # Encoder(condition=gamma_ch) -> AWGN(gamma_ch) -> Decoder(condition=gamma_ch)。
        # 提案法と同一の標準channel-noise標本を使用する。
        actual_transmitted = codec.encode(image[None], args.channel_snr_db)
        actual_symbols = real_to_complex(actual_transmitted)
        actual_received_symbols = actual_symbols + channel_std * channel_epsilon
        actual_received = complex_to_real(
            actual_received_symbols, actual_transmitted.shape
        )
        actual_channel_reconstructed = codec.decode(
            actual_received, args.channel_snr_db
        )[0]
        received_model = to_model_domain(
            received, latent_min, latent_scale, device
        )
        initial_model = to_model_domain(initial, latent_min, latent_scale, device)
        with torch.inference_mode():
            predicted_model = inverse_resfusion_from_initial(
                model, initial_model, received_model, args.channel_snr_db
            )
        predicted01 = (predicted_model + 1.0) / 2.0
        predicted_raw = predicted01 * latent_scale + latent_min
        predicted = predicted_raw.permute(0, 2, 3, 1).cpu().numpy()
        predicted = codec.power_normalize(predicted)
        reconstructed = codec.decode(predicted, high_snr)[0]

        filename = f"{path.stem}_reconstructed.png"
        output_path = reconstructed_dir / filename
        direct_low_path = direct_low_dir / filename
        direct_high_path = direct_high_dir / filename
        actual_channel_path = actual_channel_dir / filename
        Image.fromarray(np.rint(reconstructed).astype(np.uint8)).save(output_path)
        Image.fromarray(np.rint(direct_low).astype(np.uint8)).save(direct_low_path)
        Image.fromarray(np.rint(direct_high).astype(np.uint8)).save(direct_high_path)
        Image.fromarray(
            np.rint(actual_channel_reconstructed).astype(np.uint8)
        ).save(actual_channel_path)

        proposed_score = psnr(image, reconstructed)
        direct_low_score = psnr(image, direct_low)
        direct_high_score = psnr(image, direct_high)
        actual_channel_score = psnr(image, actual_channel_reconstructed)
        rows.append((
            path.name,
            str(output_path.relative_to(output_dir)), proposed_score,
            str(direct_low_path.relative_to(output_dir)), direct_low_score,
            str(direct_high_path.relative_to(output_dir)), direct_high_score,
            str(actual_channel_path.relative_to(output_dir)), actual_channel_score,
        ))
        print(
            f"[{index + 1}/{len(paths)}] {path.name}: "
            f"proposed={proposed_score:.3f}, AWGN low->low={direct_low_score:.3f}, "
            f"AWGN low->high={direct_high_score:.3f}, "
            f"actual-channel={actual_channel_score:.3f} dB"
        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((
            "input",
            "proposed_image", "proposed_psnr_db",
            "awgn_no_resfusion_low_image", "awgn_no_resfusion_low_psnr_db",
            "awgn_no_resfusion_high_image", "awgn_no_resfusion_high_psnr_db",
            "actual_channel_image", "actual_channel_psnr_db",
        ))
        writer.writerows(rows)
    routes = (
        ("awgn_noise_completion", reconstructed_dir, float(np.mean([row[2] for row in rows])), channel_std, additional_std),
        ("awgn_no_resfusion_low", direct_low_dir, float(np.mean([row[4] for row in rows])), channel_std, 0.0),
        ("awgn_no_resfusion_high", direct_high_dir, float(np.mean([row[6] for row in rows])), channel_std, 0.0),
        ("actual_channel_adjscc", actual_channel_dir, float(np.mean([row[8] for row in rows])), channel_std, 0.0),
    )
    metrics_device = torch.device(args.metrics_device)
    summary_rows = []
    print("全画像を生成しました。4経路のLPIPS、DISTS、FIDを計算します。")
    for route_name, route_dir, mean_psnr, route_channel_std, route_additional_std in routes:
        perceptual = calculate_perceptual_metrics(
            paths, route_dir, image_size, metrics_device,
            args.seed, args.metrics_batch_size,
        )
        summary_rows.append((
            route_name, args.channel_snr_db, P, start_alpha_bar,
            route_channel_std, route_additional_std, mean_psnr,
            perceptual["lpips"], perceptual["dists"], perceptual["fid"],
        ))
        print(
            f"{route_name}: PSNR={mean_psnr:.3f} dB, "
            f"LPIPS={perceptual['lpips']:.6f}, "
            f"DISTS={perceptual['dists']:.6f}, FID={perceptual['fid']:.3f}"
        )
    with (output_dir / "summary_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow((
            "route", "channel_snr_db", "P", "start_alpha_bar",
            "channel_noise_std_complex",
            "additional_noise_std_complex", "mean_psnr_db", "mean_lpips",
            "mean_dists", "fid",
        ))
        writer.writerows(summary_rows)
    print(
        f"SNR={args.channel_snr_db:g} dB, P={P:g}, alpha_bar={start_alpha_bar:.6f}, "
        f"channel_std={channel_std:.6f}, additional_std={additional_std:.6f}"
    )
    print(f"Saved: {output_dir.resolve()}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--resfusion_ckpt", required=True)
    value.add_argument("--channel_snr_db", required=True, type=float)
    value.add_argument("--adjscc_weights", default=None)
    value.add_argument("--input_dir", default="../datasets/Raindrop/test_a/gt")
    value.add_argument("--output_dir", default="my_awgn_resfusion_eval")
    value.add_argument("--low_snr", type=float, default=None)
    value.add_argument("--high_snr", type=float, default=None)
    value.add_argument("--device", default=None)
    value.add_argument("--tf_device", default=None)
    value.add_argument("--metrics_device", default="cpu")
    value.add_argument("--metrics_batch_size", type=int, default=8)
    value.add_argument("--seed", type=int, default=2024)
    value.add_argument("--limit", type=int, default=None)
    return value


if __name__ == "__main__":
    main(parser().parse_args())
