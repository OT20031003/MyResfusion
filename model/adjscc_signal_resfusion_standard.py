"""AWGN受信信号yを標準Resfusionの劣化入力とする専用Lightning module。"""

import torch
from torch.nn import functional as F

from .distributions import resfusion_x0_to_xt
from .latent_resfusion import ADJSCCSignalResfusion


class StandardReceivedADJSCCResfusion(ADJSCCSignalResfusion):
    """通常画像用Resfusionを変更せず、y -> s_highだけを学習する。"""

    def make_training_terms(self, batch):
        s_low, y, s_high, channel_snr_db = self._validate_batch(batch)
        residual = y - s_high
        t = torch.randint(0, self.T_acc, (s_high.shape[0],), device=s_high.device)
        if torch.any(t < 0) or torch.any(t >= self.T_acc):
            raise RuntimeError("diffusion timestep is outside [0, T_acc)")
        alpha_bar_t = self.alphas_hat.to(s_high.device)[t].to(s_high.dtype).reshape(-1, 1, 1, 1)
        alpha_t = self.alphas.to(s_high.device)[t].to(s_high.dtype).reshape(-1, 1, 1, 1)
        beta_t = self.betas.to(s_high.device)[t].to(s_high.dtype).reshape(-1, 1, 1, 1)
        # Datasetのchannel_epsilonとは別の乱数呼出し。分散の差引き・共有はしない。
        diffusion_epsilon = torch.randn_like(s_high)
        x_t = resfusion_x0_to_xt(s_high, alpha_bar_t, residual, diffusion_epsilon)
        # GaussianResfusion_Restoreの式24をそのまま使用する。
        residual_weight = (
            (1.0 - torch.sqrt(alpha_t)) * torch.sqrt(1.0 - alpha_bar_t) / beta_t
        )
        target_resnoise = diffusion_epsilon + residual_weight * residual
        if not all(torch.isfinite(v).all() for v in (x_t, target_resnoise, residual)):
            raise FloatingPointError("non-finite standard Resfusion training term")
        return {
            "s_low": s_low, "y": y, "s_high": s_high,
            "channel_snr_db": channel_snr_db, "residual": residual,
            "t": t, "alpha_bar_t": alpha_bar_t,
            "diffusion_epsilon": diffusion_epsilon, "x_t": x_t,
            "target_resnoise": target_resnoise,
        }

    def _loss_and_prediction(self, batch):
        terms = self.make_training_terms(batch)
        pred_resnoise = self.denoising_module(
            x=terms["x_t"], time=terms["t"], input_cond=terms["y"],
            channel_snr_db=terms["channel_snr_db"],
        )
        loss = F.mse_loss(pred_resnoise, terms["target_resnoise"])
        if not torch.isfinite(loss) or not torch.isfinite(pred_resnoise).all():
            raise FloatingPointError("non-finite loss or prediction")
        return loss, pred_resnoise, terms

    def training_step(self, batch, batch_idx: int):
        if self.mode != "epsilon" or self.loss_type != "L2":
            raise ValueError("standard ADJSCC mode requires mode=epsilon and loss_type=L2")
        loss, pred_resnoise, terms = self._loss_and_prediction(batch)
        scale = (float(self.hparams.latent_max) - float(self.hparams.latent_min)) / 2.0
        raw_s_low = (terms["s_low"] + 1.0) * scale + float(self.hparams.latent_min)
        raw_y = (terms["y"] + 1.0) * scale + float(self.hparams.latent_min)
        metrics = {
            "train_loss": loss,
            "train_t_mean": terms["t"].float().mean(),
            "train_t_min": terms["t"].float().min(),
            "train_t_max": terms["t"].float().max(),
            "train_channel_snr_db_mean": terms["channel_snr_db"].mean(),
            "train_channel_snr_db_min": terms["channel_snr_db"].min(),
            "train_channel_snr_db_max": terms["channel_snr_db"].max(),
            "train_s_low_power": 2.0 * raw_s_low.square().mean(),
            "train_y_power": 2.0 * raw_y.square().mean(),
            "train_channel_noise_power": 2.0 * (raw_y - raw_s_low).square().mean(),
            "train_residual_norm": terms["residual"].flatten(1).norm(dim=1).mean(),
            "train_x_t_mean": terms["x_t"].mean(),
            "train_x_t_std": terms["x_t"].std(unbiased=False),
            "train_pred_resnoise_norm": pred_resnoise.flatten(1).norm(dim=1).mean(),
            "train_target_resnoise_norm": terms["target_resnoise"].flatten(1).norm(dim=1).mean(),
            "train_finite": loss.new_tensor(1.0),
        }
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True,
                     prog_bar=name == "train_loss", sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        _s_low, y, s_high, channel_snr_db = self._validate_batch(batch)
        devices = [self.device.index] if self.device.type == "cuda" else []
        # AWGNだけでなくvalidation diffusion/posterior noiseもepoch間で固定する。
        with torch.random.fork_rng(devices=devices):
            validation_seed = int(self.hparams.seed) + 200000 + batch_idx
            torch.manual_seed(validation_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(validation_seed)
            val_loss, _prediction_noise, _terms = self._loss_and_prediction(batch)
            prediction = self.generate(y, channel_snr_db=channel_snr_db)
        mse = F.mse_loss(prediction, s_high)
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(mse, min=1e-12))
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True)
        self.log("val_latent_MSE", mse, on_epoch=True, sync_dist=True)
        self.log("val_latent_PSNR", psnr, on_epoch=True, prog_bar=True, sync_dist=True)
        output = {"prediction_model": prediction.detach().cpu()}
        if "image" in batch:
            output["target_image"] = batch["image"].detach().cpu()
        return output
