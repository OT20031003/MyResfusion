"""WR.tex に従う ADJSCC signal Resfusion の学習 entrypoint。"""

from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from adjscc_signal_data_wr import MINIMUM_CHANNEL_SNR_DB, WRSignalDataset
from model import WRADJSCCSignalResfusion
from model.denoising_module import RDDM_Unet
from train_adjscc_signal_resfusion import (
    ValidationImagePSNR,
    ensure_cache,
    parser as base_parser,
)
from variance_scheduler import CosineProScheduler, LinearProScheduler

def main(args):
    if args.channel_snr_min_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError(f"--channel-snr-min-dbは{MINIMUM_CHANNEL_SNR_DB:.6f} dB以上にしてください")
    if args.mode != "epsilon" or args.loss_type != "L2":
        raise ValueError("WR学習は--mode epsilon --loss_type L2専用です")
    pl.seed_everything(args.seed, workers=True)
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)
    root, cache_dir = Path(args.data_dir).expanduser(), Path(args.cache_dir).expanduser()
    train_dir = Path(args.train_gt_dir).expanduser() if args.train_gt_dir else root / "train" / "gt"
    val_dir = Path(args.val_dir).expanduser() if args.val_dir else root / "val"
    train_cache = ensure_cache(args, train_dir, cache_dir / "wr_train_pairs.pt", True, args.crops_per_image)
    val_cache = ensure_cache(args, val_dir, cache_dir / "wr_val_pairs.pt", False, 1)
    train_data = WRSignalDataset(train_cache, args.channel_snr_min_db, args.channel_snr_max_db)
    val_data = WRSignalDataset(val_cache, args.channel_snr_min_db, args.channel_snr_max_db,
                               deterministic_seed=args.seed + 100000)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=args.pin_mem, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    scheduler = LinearProScheduler(args.T) if args.noise_schedule == "LinearPro" else CosineProScheduler(args.T)
    if not torch.isclose(scheduler.get_alphas_hat()[torch.abs(torch.sqrt(scheduler.get_alphas_hat()) - .5).argmin()], torch.tensor(.25), atol=1e-4):
        raise ValueError("選択したschedulerではalpha_bar_T'=1/4になりません")
    denoiser = RDDM_Unet(dim=args.dim, out_dim=args.transmit_channel_num,
        channels=args.transmit_channel_num, input_condition=True,
        input_condition_channels=args.transmit_channel_num,
        resnet_block_groups=args.resnet_block_groups, channel_snr_condition=True,
        channel_snr_min_db=args.channel_snr_min_db, channel_snr_max_db=args.channel_snr_max_db)
    args.wr_formula_version = "WR.tex_v1"
    args.signal_domain = "raw_power_normalized_no_post_channel_normalization"
    model = WRADJSCCSignalResfusion(denoising_module=denoiser, variance_scheduler=scheduler,
        **vars(args), n_channels=args.transmit_channel_num)
    checkpoint = ModelCheckpoint(monitor="val_image_LPIPS", mode="min",
        filename="wr-best-{epoch:04d}-{val_image_LPIPS:.6f}", save_top_k=1, save_last=True,
        every_n_epochs=args.check_val_every_n_epoch)
    image_metrics = ValidationImagePSNR(
        args.adjscc_weights, args.transmit_channel_num, args.image_size,
        args.high_snr, latent_min=0.0, latent_max=1.0, seed=args.seed,
        prediction_key="prediction_raw", prediction_is_raw=True,
        power_normalize=False,
        print_metrics_table=True,
    )
    trainer = Trainer(accelerator=args.accelerator, devices=args.devices,
        max_epochs=args.epochs, accumulate_grad_batches=args.accum_iter,
        default_root_dir=args.log_dir, check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip, precision=args.precision,
        callbacks=[image_metrics, checkpoint, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=1, deterministic="warn",
        strategy="auto" if args.devices == 1 else "ddp")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_ckpt)

def parser():
    value = base_parser()
    value.description = __doc__
    value.set_defaults(cache_dir="wr_signal_cache", log_dir="wr_signal_train",
                       channel_snr_min_db=MINIMUM_CHANNEL_SNR_DB)
    return value

if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments._build_cache:
        from train_adjscc_signal_resfusion import build_requested_cache
        build_requested_cache(arguments)
    else:
        main(arguments)
