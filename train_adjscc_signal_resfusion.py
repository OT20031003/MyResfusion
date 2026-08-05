"""low-SNR ADJSCC信号からhigh-SNR信号へ変換するResfusionを学習する。"""

import argparse
import subprocess
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from adjscc_signal_data import (
    SIGNAL_PROCESSING_VERSION,
    SignalPairDataset,
    build_pair_cache,
    checkpoint_fingerprint,
    latent_range,
    load_cache,
    validate_cache,
)
from model import LatentResfusion
from model.denoising_module import RDDM_Unet
from variance_scheduler import CosineProScheduler, LinearProScheduler


def cache_expectation(args, image_dir: Path, random_crop: bool, crops: int) -> dict:
    return {
        "signal_processing": SIGNAL_PROCESSING_VERSION,
        "image_dir": str(image_dir.resolve()),
        "adjscc_weights": str(Path(args.adjscc_weights).expanduser().resolve()),
        "adjscc_fingerprint": checkpoint_fingerprint(args.adjscc_weights),
        "low_snr": float(args.low_snr),
        "high_snr": float(args.high_snr),
        "transmit_channel_num": args.transmit_channel_num,
        "image_size": args.image_size,
        "crops_per_image": crops,
        "random_crop": random_crop,
        "seed": args.seed,
    }


def build_requested_cache(args) -> None:
    build_pair_cache(
        image_dir=args._cache_image_dir,
        output_path=args._cache_output,
        adjscc_weights=args.adjscc_weights,
        low_snr=args.low_snr,
        high_snr=args.high_snr,
        transmit_channel_num=args.transmit_channel_num,
        batch_size=args.encoder_batch_size,
        image_size=args.image_size,
        crops_per_image=args._cache_crops,
        random_crop=args._cache_random_crop,
        seed=args.seed,
        tf_device=args.tf_device,
    )


def ensure_cache(args, image_dir: Path, output: Path, random_crop: bool, crops: int) -> dict:
    expected = cache_expectation(args, image_dir, random_crop, crops)
    needs_build = args.rebuild_cache or not output.is_file()
    if not needs_build:
        cache = load_cache(str(output))
        validate_cache(cache, expected)
        return cache

    # TensorFlowが確保したGPUメモリを学習開始前に確実に解放するため別プロセスにする。
    command = [
        sys.executable, str(Path(__file__).resolve()), "--_build_cache",
        "--_cache_image_dir", str(image_dir), "--_cache_output", str(output),
        "--_cache_crops", str(crops), "--adjscc_weights", args.adjscc_weights,
        "--low_snr", str(args.low_snr), "--high_snr", str(args.high_snr),
        "--transmit_channel_num", str(args.transmit_channel_num),
        "--encoder_batch_size", str(args.encoder_batch_size),
        "--image_size", str(args.image_size), "--seed", str(args.seed),
    ]
    if random_crop:
        command.append("--_cache_random_crop")
    if args.tf_device:
        command.extend(["--tf_device", args.tf_device])
    subprocess.run(command, check=True)
    cache = load_cache(str(output))
    validate_cache(cache, expected)
    return cache


def main(args) -> None:
    if args.low_snr >= args.high_snr:
        raise ValueError("--low_snrは--high_snrより小さくしてください")
    pl.seed_everything(args.seed, workers=True)
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)

    data_root = Path(args.data_dir).expanduser()
    train_dir = Path(args.train_gt_dir).expanduser() if args.train_gt_dir else data_root / "train" / "gt"
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else data_root / "val"
    cache_dir = Path(args.cache_dir).expanduser()
    train_path = cache_dir / "train_signal_pairs.pt"
    val_path = cache_dir / "val_signal_pairs.pt"
    train_cache = ensure_cache(args, train_dir, train_path, True, args.crops_per_image)
    val_cache = ensure_cache(args, val_dir, val_path, False, 1)

    latent_min, latent_max = latent_range(train_cache)
    train_dataset = SignalPairDataset(train_cache, latent_min, latent_max)
    val_dataset = SignalPairDataset(val_cache, latent_min, latent_max)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, persistent_workers=args.num_workers > 0,
    )

    scheduler = LinearProScheduler(args.T) if args.noise_schedule == "LinearPro" else CosineProScheduler(args.T)
    denoiser = RDDM_Unet(
        dim=args.dim, out_dim=args.transmit_channel_num, channels=args.transmit_channel_num,
        input_condition=True, input_condition_channels=args.transmit_channel_num,
        resnet_block_groups=args.resnet_block_groups,
    )
    # 復元時に必要なADJSCC条件と正規化範囲もLightning checkpointへ格納する。
    model = LatentResfusion(
        denoising_module=denoiser, variance_scheduler=scheduler,
        **vars(args), n_channels=args.transmit_channel_num,
        latent_min=latent_min, latent_max=latent_max,
    )
    checkpoint = ModelCheckpoint(
        monitor="val_latent_PSNR", mode="max",
        filename="best-{epoch:04d}-{val_latent_PSNR:.3f}", save_top_k=1, save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
    )
    trainer = Trainer(
        accelerator=args.accelerator, devices=args.devices, max_epochs=args.epochs,
        accumulate_grad_batches=args.accum_iter, default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip, precision=args.precision,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=1, deterministic="warn",
        strategy="auto" if args.devices == 1 else "ddp",
    )
    print(f"Train pairs: {len(train_dataset)}, validation pairs: {len(val_dataset)}")
    print(f"Signal shape: {tuple(train_cache['low'].shape[1:])} (NCHW)")
    print(f"Shared latent range: [{latent_min:.6g}, {latent_max:.6g}]")
    print("Channel: disabled")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_ckpt)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data_dir", default="../datasets/Raindrop")
    value.add_argument("--train_gt_dir", default=None)
    value.add_argument("--val_dir", default=None)
    value.add_argument("--adjscc_weights", required=True)
    value.add_argument("--low_snr", type=float, default=-10.0)
    value.add_argument("--high_snr", type=float, default=20.0)
    value.add_argument("--transmit_channel_num", type=int, default=16)
    value.add_argument("--image_size", type=int, default=256)
    value.add_argument("--cache_dir", default="my_resfusion_cache")
    value.add_argument("--crops_per_image", type=int, default=1)
    value.add_argument("--encoder_batch_size", type=int, default=4)
    value.add_argument("--tf_device", default=None, help="例: /GPU:0 または /CPU:0")
    value.add_argument("--rebuild_cache", action="store_true")
    value.add_argument("--epochs", type=int, default=500)
    value.add_argument("--batch_size", type=int, default=4)
    value.add_argument("--num_workers", type=int, default=4)
    value.add_argument("--pin_mem", dest="pin_mem", action="store_true")
    value.add_argument("--no_pin_mem", dest="pin_mem", action="store_false")
    value.set_defaults(pin_mem=True)
    value.add_argument("--check_val_every_n_epoch", type=int, default=1)
    value.add_argument("--accum_iter", type=int, default=1)
    value.add_argument("--gradient_clip", type=float, default=1.0)
    value.add_argument("--precision", default="32")
    value.add_argument("--seed", type=int, default=2024)
    value.add_argument("--noise_schedule", choices=("LinearPro", "CosinePro"), default="LinearPro")
    value.add_argument("--T", type=int, default=12)
    value.add_argument("--mode", choices=("epsilon", "sample", "residual"), default="epsilon")
    value.add_argument("--loss_type", choices=("L1", "L2"), default="L2")
    value.add_argument("--optimizer_type", choices=("Adam", "AdamW", "SGD"), default="AdamW")
    value.add_argument("--lr_scheduler_type", choices=("CosineAnnealingLR", "ReduceLROnPlateau"), default="CosineAnnealingLR")
    value.add_argument("--dim", type=int, default=64)
    value.add_argument("--resnet_block_groups", type=int, default=8)
    value.add_argument("--blr", type=float, default=8.8e-4)
    value.add_argument("--min_lr", type=float, default=3e-5)
    value.add_argument("--weight_decay", type=float, default=0.0)
    value.add_argument("--accelerator", default="gpu")
    value.add_argument("--devices", type=int, default=1)
    value.add_argument("--num_nodes", type=int, default=1)
    value.add_argument("--log_dir", default="my_resfusion_train")
    value.add_argument("--resume_ckpt", default=None)
    value.add_argument("--matmul_precision", choices=("default", "medium", "high"), default="high")
    value.add_argument("--_build_cache", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--_cache_image_dir", default=None, help=argparse.SUPPRESS)
    value.add_argument("--_cache_output", default=None, help=argparse.SUPPRESS)
    value.add_argument("--_cache_crops", type=int, default=1, help=argparse.SUPPRESS)
    value.add_argument("--_cache_random_crop", action="store_true", help=argparse.SUPPRESS)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments._build_cache:
        build_requested_cache(arguments)
    else:
        main(arguments)
