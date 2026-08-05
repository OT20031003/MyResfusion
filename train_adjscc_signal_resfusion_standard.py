"""AWGN受信信号yからs_highを復元する標準Resfusionを新規学習する。"""

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from adjscc_signal_data import latent_range
from adjscc_signal_data_standard import (
    MINIMUM_CHANNEL_SNR_DB,
    SIGNAL_PROCESSING_VERSION,
    StandardReceivedSignalDataset,
)
from model.adjscc_signal_resfusion_standard import StandardReceivedADJSCCResfusion
from model.denoising_module import RDDM_Unet
from train_adjscc_signal_resfusion import ValidationImagePSNR, ensure_cache, parser as base_parser
from variance_scheduler import CosineProScheduler, LinearProScheduler


def main(args) -> None:
    if args.low_snr >= args.high_snr:
        raise ValueError("--low_snr must be lower than --high_snr")
    if args.channel_snr_min_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError("--channel-snr-min-db must be at least -4 dB")
    if args.channel_snr_max_db < args.channel_snr_min_db:
        raise ValueError("--channel-snr-max-db must be >= --channel-snr-min-db")
    if args.mode != "epsilon" or args.loss_type != "L2":
        raise ValueError("standard received-y training requires --mode epsilon --loss_type L2")
    pl.seed_everything(args.seed, workers=True)
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)

    data_root = Path(args.data_dir).expanduser()
    train_dir = Path(args.train_gt_dir).expanduser() if args.train_gt_dir else data_root / "train" / "gt"
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else data_root / "val"
    cache_dir = Path(args.cache_dir).expanduser()
    # 旧残差定義のcache/output名と混同しない。ただしcache内のclean signal pairは再利用可能。
    train_cache = ensure_cache(
        args, train_dir, cache_dir / "standard_received_y_train_pairs.pt",
        True, args.crops_per_image,
    )
    val_cache = ensure_cache(
        args, val_dir, cache_dir / "standard_received_y_val_pairs.pt", False, 1
    )
    latent_min, latent_max = latent_range(train_cache)
    train_dataset = StandardReceivedSignalDataset(
        train_cache, latent_min, latent_max,
        args.channel_snr_min_db, args.channel_snr_max_db,
    )
    val_dataset = StandardReceivedSignalDataset(
        val_cache, latent_min, latent_max,
        args.channel_snr_min_db, args.channel_snr_max_db,
        deterministic_seed=args.seed + 100000,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
        persistent_workers=args.num_workers > 0,
    )

    scheduler = (
        LinearProScheduler(args.T) if args.noise_schedule == "LinearPro"
        else CosineProScheduler(args.T)
    )
    denoiser = RDDM_Unet(
        dim=args.dim, out_dim=args.transmit_channel_num,
        channels=args.transmit_channel_num, input_condition=True,
        input_condition_channels=args.transmit_channel_num,
        resnet_block_groups=args.resnet_block_groups,
        channel_snr_condition=True,
        channel_snr_min_db=args.channel_snr_min_db,
        channel_snr_max_db=args.channel_snr_max_db,
    )
    # Lightning hyper_parametersへ新方式を識別するmetadataを保存する。
    args.degraded_input_mode = "received_y"
    args.residual_definition = "y_minus_s_high"
    args.channel_noise_in_diffusion = False
    args.standard_resfusion_initialization = True
    args.signal_processing_version = SIGNAL_PROCESSING_VERSION
    model = StandardReceivedADJSCCResfusion(
        denoising_module=denoiser, variance_scheduler=scheduler,
        **vars(args), n_channels=args.transmit_channel_num,
        latent_min=latent_min, latent_max=latent_max,
    )

    monitor_mode = "min" if args.checkpoint_monitor in {"val_loss", "val_latent_MSE"} else "max"
    checkpoint = ModelCheckpoint(
        monitor=args.checkpoint_monitor, mode=monitor_mode,
        filename="standard-best-{epoch:04d}-{val_image_PSNR:.3f}",
        save_top_k=1, save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
    )
    image_metrics = ValidationImagePSNR(
        args.adjscc_weights, args.transmit_channel_num, args.image_size,
        args.high_snr, latent_min, latent_max, args.seed,
    )
    trainer = Trainer(
        accelerator=args.accelerator, devices=args.devices,
        max_epochs=args.epochs, accumulate_grad_batches=args.accum_iter,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip, precision=args.precision,
        callbacks=[image_metrics, checkpoint, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=1, deterministic="warn",
        strategy="auto" if args.devices == 1 else "ddp",
    )
    print(f"Standard received-y train/val: {len(train_dataset)}/{len(val_dataset)}")
    print(f"Channel SNR: [{args.channel_snr_min_db:g}, {args.channel_snr_max_db:g}] dB")
    print(f"Checkpoint monitor: {args.checkpoint_monitor} ({monitor_mode})")
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader,
        ckpt_path=args.resume_ckpt,
    )


def parser():
    value = base_parser()
    value.description = __doc__
    value.add_argument(
        "--checkpoint-monitor",
        choices=("val_image_PSNR", "val_latent_PSNR", "val_latent_MSE", "val_loss"),
        default="val_image_PSNR",
    )
    value.set_defaults(
        cache_dir="standard_received_y_cache",
        log_dir="standard_received_y_train",
    )
    return value


if __name__ == "__main__":
    main(parser().parse_args())
