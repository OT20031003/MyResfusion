"""WR.tex の学習式と推論式を実装する Lightning module。"""

import torch
from torch.nn import functional as F
from .latent_resfusion import ADJSCCSignalResfusion

class WRADJSCCSignalResfusion(ADJSCCSignalResfusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # WR.tex の明示条件 alpha_bar_0=1。配列 index 0 は paper t=1。
        self.alphas_hat_t_minus_1 = self.alphas_hat_t_minus_1.clone()
        self.betas_hat = self.betas_hat.clone()
        self.alphas_hat_t_minus_1[0] = 1.0
        self.betas_hat[0] = 0.0

    def _validate_wr_batch(self, batch):
        required = {"z_low", "z_high", "y", "channel_noise", "sigma", "channel_snr_db"}
        if not isinstance(batch, dict) or not required.issubset(batch):
            raise ValueError(f"WR batchには{sorted(required)}が必要です")
        z_low, z_high, y = batch["z_low"], batch["z_high"], batch["y"]
        if z_low.ndim != 4 or z_low.shape != z_high.shape or z_low.shape != y.shape:
            raise ValueError("z_low, z_high, yは同一shapeのNCHW tensorにしてください")
        return z_low, z_high, y

    def make_training_terms(self, batch):
        z_low, z_high, y = self._validate_wr_batch(batch)
        # index 0 が論文の t=1 に対応する。
        t = torch.randint(0, self.T_acc, (len(z_high),), device=z_high.device)
        shape = (-1, 1, 1, 1)
        alpha_bar = self.alphas_hat.to(z_high.device)[t].to(z_high.dtype).reshape(shape)
        alpha = self.alphas.to(z_high.device)[t].to(z_high.dtype).reshape(shape)
        beta = self.betas.to(z_high.device)[t].to(z_high.dtype).reshape(shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sigma = batch["sigma"].to(z_high).reshape(shape)
        n = batch["channel_noise"].to(z_high)
        epsilon_add = torch.randn_like(z_high)
        completion_variance = 1.0 - alpha_bar - (1.0 - sqrt_alpha_bar).square() * sigma.square()
        if torch.any(completion_variance < -1e-6):
            raise ValueError("WR式の追加noise分散が負です")
        completion_std = torch.sqrt(torch.clamp(completion_variance, min=0.0))
        z_t = ((2.0 * sqrt_alpha_bar - 1.0) * z_high
               + (1.0 - sqrt_alpha_bar) * y + completion_std * epsilon_add)
        epsilon = ((1.0 - sqrt_alpha_bar) * sigma * n
                   + completion_std * epsilon_add) / torch.sqrt(1.0 - alpha_bar)
        residual = z_low - z_high
        residual_weight = (1.0 - torch.sqrt(alpha)) * torch.sqrt(1.0 - alpha_bar) / beta
        target = epsilon + residual_weight * residual
        if not all(torch.isfinite(x).all() for x in (z_t, epsilon, target)):
            raise FloatingPointError("WR training termにNaN/Infがあります")
        return {"z_low": z_low, "z_high": z_high, "y": y, "t": t,
                "alpha_bar": alpha_bar, "epsilon_add": epsilon_add,
                "completion_variance": completion_variance, "epsilon": epsilon,
                "residual": residual, "z_t": z_t, "target_resnoise": target}

    def _loss_and_prediction(self, batch):
        terms = self.make_training_terms(batch)
        prediction = self.denoising_module(x=terms["z_t"], time=terms["t"],
            input_cond=terms["y"], channel_snr_db=batch["channel_snr_db"].to(terms["z_t"]))
        return F.mse_loss(prediction, terms["target_resnoise"]), prediction, terms

    def training_step(self, batch, batch_idx):
        if self.mode != "epsilon" or self.loss_type != "L2":
            raise ValueError("WR.texはmode=epsilon, loss_type=L2専用です")
        loss, _, terms = self._loss_and_prediction(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_t_paper", terms["t"].float().mean() + 1, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._loss_and_prediction(batch)
        generated = self.infer_from_received(batch["y"], batch["sigma"], batch["channel_snr_db"])
        mse = F.mse_loss(generated, batch["z_high"])
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_latent_MSE", mse, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_channel_snr_db", batch["channel_snr_db"].float().mean(),
                 on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 sync_dist=True, batch_size=len(batch["z_high"]))
        return {"prediction_raw": generated.detach().cpu(),
                "target_image": batch["image"].cpu(),
                "val_loss": loss.detach().cpu(), "val_latent_MSE": mse.detach().cpu()}

    def infer_from_received(self, y, sigma, channel_snr_db, stochastic=True):
        start_alpha_bar = self.alphas_hat[self.T_acc - 1].to(y)
        if not torch.isclose(start_alpha_bar, y.new_tensor(0.25), atol=1e-4):
            raise ValueError("WR推論にはalpha_bar_T'=1/4のschedulerが必要です")
        sigma = sigma.to(y).reshape(-1, 1, 1, 1)
        extra_var = (3.0 - sigma.square()) / 4.0
        if torch.any(extra_var < -1e-6):
            raise ValueError("WR推論の追加noise分散が負です")
        state = 0.5 * y + torch.sqrt(torch.clamp(extra_var, min=0.0)) * torch.randn_like(y)
        snr = channel_snr_db.to(y).reshape(-1)
        for index in range(self.T_acc - 1, -1, -1):
            time = torch.full((len(y),), index, dtype=torch.long, device=y.device)
            pred = self.denoising_module(x=state, time=time, input_cond=y, channel_snr_db=snr)
            posterior = torch.randn_like(state) if stochastic and index > 0 else torch.zeros_like(state)
            state = ((state - self.betas[index] / torch.sqrt(1.0 - self.alphas_hat[index]) * pred)
                     / torch.sqrt(self.alphas[index]) + torch.sqrt(self.betas_hat[index]) * posterior)
        return state
