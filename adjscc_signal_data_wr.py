"""WR.tex 専用の ADJSCC 信号 dataset（受信後の正規化・clipなし）。"""

from typing import Dict, Optional
import torch
from torch.utils.data import Dataset

MINIMUM_CHANNEL_SNR_DB = -10.0 * torch.log10(torch.tensor(3.0)).item()

def sigma_from_snr_db(snr_db: torch.Tensor) -> torch.Tensor:
    return torch.pow(snr_db.new_tensor(10.0), -snr_db / 20.0)

class WRSignalDataset(Dataset):
    def __init__(self, cache: Dict, channel_snr_min_db: float,
                 channel_snr_max_db: float, deterministic_seed: Optional[int] = None):
        if channel_snr_min_db < MINIMUM_CHANNEL_SNR_DB:
            raise ValueError(f"WRのsigma^2<=3よりSNRは{MINIMUM_CHANNEL_SNR_DB:.6f} dB以上です")
        if channel_snr_max_db < channel_snr_min_db:
            raise ValueError("channel_snr_max_dbはchannel_snr_min_db以上にしてください")
        self.low, self.high, self.images = cache["low"], cache["high"], cache["images"]
        self.snr_min, self.snr_max = float(channel_snr_min_db), float(channel_snr_max_db)
        self.deterministic_seed = deterministic_seed

    def __len__(self):
        return len(self.low)

    def __getitem__(self, index):
        z_low, z_high = self.low[index].float(), self.high[index].float()
        generator = None if self.deterministic_seed is None else torch.Generator().manual_seed(self.deterministic_seed + index)
        if self.snr_min == self.snr_max:
            snr_db = z_low.new_tensor(self.snr_min)
        else:
            snr_db = torch.empty((), dtype=z_low.dtype).uniform_(self.snr_min, self.snr_max, generator=generator)
        n = torch.randn(z_low.shape, dtype=z_low.dtype, generator=generator)
        sigma = sigma_from_snr_db(snr_db)
        y = z_low + sigma * n
        return {"z_low": z_low, "z_high": z_high, "y": y, "channel_noise": n,
                "sigma": sigma, "channel_snr_db": snr_db, "image": self.images[index]}
