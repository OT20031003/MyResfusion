"""AWGN受信信号yを劣化入力とする標準Resfusion用データ処理。"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SIGNAL_PROCESSING_VERSION = "awgn_received_y_standard_resfusion_v1"
MINIMUM_CHANNEL_SNR_DB = -4.0


def to_model_domain(
    signal_raw: torch.Tensor, latent_min: float, latent_max: float
) -> torch.Tensor:
    """固定train-cache統計で物理信号をmodel-domainへ写像する。clipしない。"""
    if latent_max <= latent_min:
        raise ValueError("latent_max must be greater than latent_min")
    return 2.0 * (signal_raw.float() - latent_min) / (latent_max - latent_min) - 1.0


def from_model_domain(
    signal_model: torch.Tensor, latent_min: float, latent_max: float
) -> torch.Tensor:
    """to_model_domainの逆変換。train/test共通で使用する。"""
    if latent_max <= latent_min:
        raise ValueError("latent_max must be greater than latent_min")
    return (signal_model.float() + 1.0) * (latent_max - latent_min) / 2.0 + latent_min


def sample_channel_awgn(
    s_low_raw: torch.Tensor,
    channel_snr_db: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CHW実数表現へ複素AWGNを加える。返値は(y, epsilon, noise)。"""
    if s_low_raw.ndim != 3:
        raise ValueError(f"s_low_raw must be CHW, got {tuple(s_low_raw.shape)}")
    if not torch.isfinite(channel_snr_db) or channel_snr_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError("channel_snr_db must be at least -4 dB")
    # E[|channel_epsilon|^2]=1: 実部・虚部の各分散は1/2。
    channel_epsilon = torch.randn(
        s_low_raw.shape, dtype=s_low_raw.dtype, device=s_low_raw.device,
        generator=generator,
    ) / np.sqrt(2.0)
    channel_sigma = torch.pow(
        channel_snr_db.new_tensor(10.0), -channel_snr_db / 20.0
    )
    channel_noise = channel_sigma * channel_epsilon
    y_raw = s_low_raw + channel_noise
    return y_raw, channel_epsilon, channel_noise


class StandardReceivedSignalDataset(Dataset):
    """送信前正規化済みcacheから、sampleごとにAWGN受信信号を生成する。"""

    def __init__(
        self, cache: Dict, latent_min: float, latent_max: float,
        channel_snr_min_db: float, channel_snr_max_db: float,
        deterministic_seed: Optional[int] = None,
    ) -> None:
        if channel_snr_min_db < MINIMUM_CHANNEL_SNR_DB:
            raise ValueError("channel_snr_min_db must be at least -4 dB")
        if channel_snr_max_db < channel_snr_min_db:
            raise ValueError("channel_snr_max_db must be >= channel_snr_min_db")
        self.low = cache["low"]
        self.high = cache["high"]
        self.images = cache.get("images")
        self.latent_min = float(latent_min)
        self.latent_max = float(latent_max)
        self.channel_snr_min_db = float(channel_snr_min_db)
        self.channel_snr_max_db = float(channel_snr_max_db)
        self.deterministic_seed = deterministic_seed

    def __len__(self) -> int:
        return len(self.low)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        s_low_raw = self.low[index].float()       # [C,H,W], power-normalized before AWGN
        s_high_raw = self.high[index].float()     # [C,H,W], power-normalized
        if s_low_raw.ndim != 3 or s_low_raw.shape != s_high_raw.shape:
            raise RuntimeError("cached low/high signals must have identical CHW shape")
        generator = None
        if self.deterministic_seed is not None:
            generator = torch.Generator().manual_seed(self.deterministic_seed + index)
        if self.channel_snr_min_db == self.channel_snr_max_db:
            channel_snr_db = s_low_raw.new_tensor(self.channel_snr_min_db)
        else:
            channel_snr_db = torch.empty((), dtype=s_low_raw.dtype).uniform_(
                self.channel_snr_min_db, self.channel_snr_max_db,
                generator=generator,
            )
        # y生成後にpower normalization、clamp、sample別rescaleは行わない。
        y_raw, channel_epsilon, channel_noise = sample_channel_awgn(
            s_low_raw, channel_snr_db, generator
        )
        if not all(torch.isfinite(v).all() for v in (s_low_raw, s_high_raw, y_raw)):
            raise FloatingPointError("non-finite ADJSCC signal")
        batch = {
            "s_low": to_model_domain(s_low_raw, self.latent_min, self.latent_max),
            "y": to_model_domain(y_raw, self.latent_min, self.latent_max),
            "s_high": to_model_domain(s_high_raw, self.latent_min, self.latent_max),
            "channel_snr_db": channel_snr_db,
            # debug/test用。学習modelはchannel noiseを拡散noiseへ利用しない。
            "channel_epsilon": channel_epsilon,
            "channel_noise": channel_noise,
        }
        if self.images is not None:
            batch["image"] = self.images[index]
        return batch
