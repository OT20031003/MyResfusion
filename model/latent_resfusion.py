"""多チャネルADJSCC信号用のResfusion Lightning module。"""

import torch
from torch.nn import functional as F

from .distributions import resfusion_x0_to_xt
from .resfusion_restore import GaussianResfusion_Restore


class LatentResfusion(GaussianResfusion_Restore):
    """画像表示を行わず、信号空間の検証値を記録するResfusion。"""

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch
        prediction = self.generate(inputs * 2.0 - 1.0)
        prediction = torch.clamp((prediction + 1.0) / 2.0, 0.0, 1.0)
        mse = F.mse_loss(prediction, targets)
        psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
        self.log("val_latent_mse", mse, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_latent_PSNR", psnr, on_epoch=True, prog_bar=True, sync_dist=True)
        # 画像decodeはtrain scriptのcallbackでCPU上にて行う。
        return {
            "prediction_model": prediction.detach().cpu(),
            "target_image": batch["image"].detach().cpu(),
        }

    def test_step(self, batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)


class ADJSCCSignalResfusion(LatentResfusion):
    """clean s_lowを残差anchor、AWGN付きyを条件とする信号専用Resfusion。"""

    def _validate_batch(self, batch):
        required = {"s_low", "y", "s_high", "channel_snr_db"}
        if not isinstance(batch, dict) or not required.issubset(batch):
            raise ValueError(f"ADJSCC batchには{sorted(required)}が必要です")
        s_low, y, s_high = batch["s_low"], batch["y"], batch["s_high"]
        if s_low.ndim != 4 or s_low.shape != s_high.shape or s_low.shape != y.shape:
            raise ValueError("s_low, s_high, yは同一shapeのNCHW tensorにしてください")
        channel_snr_db = batch["channel_snr_db"].to(device=s_low.device, dtype=s_low.dtype)
        channel_snr_db = channel_snr_db.reshape(-1)
        if channel_snr_db.shape[0] != s_low.shape[0]:
            raise ValueError("channel_snr_dbはsampleごとに1値必要です")
        if torch.any(channel_snr_db < -4.0):
            raise ValueError("channel_snr_dbは-4 dB以上にしてください")
        if not all(torch.isfinite(value).all() for value in (s_low, y, s_high, channel_snr_db)):
            raise FloatingPointError("ADJSCC batchにNaN/Infがあります")
        return s_low, y, s_high, channel_snr_db

    def make_training_terms(self, batch):
        """学習項を構成する。全信号は固定変換後のNCHW、SNRは[B] dB。"""
        s_low, y, s_high, channel_snr_db = self._validate_batch(batch)
        residual = s_low - s_high  # yではなくcleanな送信信号をanchorにする。
        t = torch.randint(0, self.T_acc, (s_high.shape[0],), device=s_high.device)
        if torch.any(t < 0) or torch.any(t >= self.T_acc):
            raise RuntimeError("diffusion timestepがT_acc範囲外です")
        alpha_hat = self.alphas_hat.to(s_high.device)[t].to(s_high.dtype).reshape(-1, 1, 1, 1)
        # 通信路AWGNとは別の呼出しで、独立な拡散noiseを生成する。
        eps_diff = torch.randn_like(s_high)
        x_t = resfusion_x0_to_xt(s_high, alpha_hat, residual, eps_diff)
        alpha = self.alphas.to(s_high.device)[t].to(s_high.dtype).reshape(-1, 1, 1, 1)
        beta = self.betas.to(s_high.device)[t].to(s_high.dtype).reshape(-1, 1, 1, 1)
        # GaussianResfusion_Restore.training_step（論文式24）と同一の係数式。
        residual_weight = (1.0 - torch.sqrt(alpha)) * torch.sqrt(1.0 - alpha_hat) / beta
        target_resnoise = eps_diff + residual_weight * residual
        if x_t.shape != s_low.shape or not torch.isfinite(x_t).all() \
                or not torch.isfinite(target_resnoise).all():
            raise FloatingPointError("x_tまたはresnoise教師信号が不正です")
        return {
            "s_low": s_low, "y": y, "s_high": s_high,
            "channel_snr_db": channel_snr_db, "residual": residual,
            "t": t, "eps_diff": eps_diff, "x_t": x_t,
            "target_resnoise": target_resnoise,
        }

    def training_step(self, batch, batch_idx: int):
        if self.mode != "epsilon":
            raise ValueError("ADJSCCSignalResfusionはmode=epsilonのみを対象とします")
        terms = self.make_training_terms(batch)
        pred_resnoise = self.denoising_module(
            x=terms["x_t"], time=terms["t"], input_cond=terms["y"],
            channel_snr_db=terms["channel_snr_db"],
        )
        if not torch.isfinite(pred_resnoise).all():
            raise FloatingPointError("pred_resnoiseにNaN/Infがあります")
        if self.loss_type == "L2":
            loss = F.mse_loss(pred_resnoise, terms["target_resnoise"])
        elif self.loss_type == "L1":
            loss = F.smooth_l1_loss(pred_resnoise, terms["target_resnoise"])
        else:
            raise ValueError("Wrong loss type !!!")
        if not torch.isfinite(loss):
            raise FloatingPointError("resnoise lossがNaN/Infです")

        # model-domain差から物理信号差へ戻す。固定affineなのでnoiseにはoffsetがない。
        raw_per_model_unit = (float(self.hparams.latent_max) - float(self.hparams.latent_min)) / 2.0
        raw_s_low = (terms["s_low"] + 1.0) * raw_per_model_unit + float(self.hparams.latent_min)
        raw_y = (terms["y"] + 1.0) * raw_per_model_unit + float(self.hparams.latent_min)
        metrics = {
            "train_loss": loss,
            "train_t_mean": terms["t"].float().mean(),
            "train_t_min": terms["t"].min().float(),
            "train_t_max": terms["t"].max().float(),
            "train_channel_snr_db_mean": terms["channel_snr_db"].mean(),
            "train_channel_snr_db_min": terms["channel_snr_db"].min(),
            "train_channel_snr_db_max": terms["channel_snr_db"].max(),
            # 実部・虚部を同数保持する実数tensorなので、複素symbol平均電力は2倍。
            "train_s_low_power": 2.0 * raw_s_low.square().mean(),
            "train_y_power": 2.0 * raw_y.square().mean(),
            "train_channel_noise_power": 2.0 * (raw_y - raw_s_low).square().mean(),
            "train_x_t_mean": terms["x_t"].mean(),
            "train_x_t_std": terms["x_t"].std(unbiased=False),
            "train_residual_norm": terms["residual"].flatten(1).norm(dim=1).mean(),
            "train_pred_resnoise_norm": pred_resnoise.flatten(1).norm(dim=1).mean(),
            "train_target_resnoise_norm": terms["target_resnoise"].flatten(1).norm(dim=1).mean(),
        }
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True,
                     prog_bar=name == "train_loss", logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        _s_low, y, s_high, channel_snr_db = self._validate_batch(batch)
        prediction = self.generate(y, channel_snr_db=channel_snr_db)
        mse = F.mse_loss(prediction, s_high)
        # 固定affine座標でのPSNR。data rangeは学習cacheのraw rangeに対応する2。
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(mse, min=1e-12))
        self.log("val_latent_mse", mse, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_latent_PSNR", psnr, on_epoch=True, prog_bar=True, sync_dist=True)
