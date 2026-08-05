"""多チャネルADJSCC信号用のResfusion Lightning module。"""

import torch
from torch.nn import functional as F

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

    def test_step(self, batch, batch_idx: int):
        return self.validation_step(batch, batch_idx)
