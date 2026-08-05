"""ADJSCC信号ペアの作成・保存・読込ユーティリティ。"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SIGNAL_PROCESSING_VERSION = "power_normalized_no_awgn_with_images_v2"
MINIMUM_CHANNEL_SNR_DB = -4.0


def model_domain(signal: torch.Tensor, latent_min: float, latent_max: float) -> torch.Tensor:
    """固定cache統計量でraw信号を[-1, 1]座標へ写像する（clipしない）。"""
    return 2.0 * (signal.float() - latent_min) / (latent_max - latent_min) - 1.0


def add_channel_awgn(
    s_low: torch.Tensor, channel_snr_db: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """電力正規化済み実数表現へ、複素AWGNを加える。

    Args:
        s_low: 送信前の物理信号 [C,H,W]。前半/後半が複素信号の実部/虚部。
        channel_snr_db: scalar tensor、単位dB。
    Returns:
        (y, eps_channel): ともに[C,H,W]。yへの再正規化・clampは行わない。
    """
    if s_low.ndim != 3:
        raise ValueError(f"s_lowはCHW形式にしてください: shape={tuple(s_low.shape)}")
    if not torch.isfinite(channel_snr_db) or channel_snr_db < MINIMUM_CHANNEL_SNR_DB:
        raise ValueError(f"channel SNRは{MINIMUM_CHANNEL_SNR_DB:g} dB以上にしてください")
    # 複素標準Gaussian E[|eps|^2]=1 なので、各実数成分の分散は1/2。
    eps_channel = torch.randn(
        s_low.shape, dtype=s_low.dtype, device=s_low.device, generator=generator
    ) / np.sqrt(2.0)
    sigma_channel = torch.pow(
        channel_snr_db.new_tensor(10.0), -channel_snr_db / 20.0
    )
    return s_low + sigma_channel * eps_channel, eps_channel


def list_images(directory: str) -> List[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"画像ディレクトリが見つかりません: {root}")
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"画像がありません: {root}")
    return paths


def load_crop(path: Path, image_size: int, random_crop: bool, rng: np.random.Generator) -> np.ndarray:
    """ResfusionのRaindrop処理に合わせ、256角cropと左右反転を行う。"""
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width < image_size or height < image_size:
            scale = max(image_size / width, image_size / height)
            image = image.resize(
                (int(round(width * scale)), int(round(height * scale))), Image.Resampling.BICUBIC
            )
            width, height = image.size
        if random_crop:
            left = int(rng.integers(0, width - image_size + 1))
            top = int(rng.integers(0, height - image_size + 1))
        else:
            left = (width - image_size) // 2
            top = (height - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))
        if random_crop and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return np.asarray(image, dtype=np.float32)


def checkpoint_fingerprint(path: str) -> str:
    checkpoint = Path(path).expanduser().resolve()
    stat = checkpoint.stat()
    value = f"{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def build_pair_cache(
    image_dir: str,
    output_path: str,
    adjscc_weights: str,
    low_snr: float,
    high_snr: float,
    transmit_channel_num: int,
    batch_size: int,
    image_size: int = 256,
    crops_per_image: int = 1,
    random_crop: bool = True,
    seed: int = 2024,
    tf_device: Optional[str] = None,
) -> None:
    """同じcropからlow/high条件の電力正規化済み・AWGNなし信号を作る。"""
    from ADJSCC.adjscc_module import ADJSCCCodec

    if crops_per_image < 1 or batch_size < 1:
        raise ValueError("crops_per_imageとbatch_sizeは1以上にしてください")
    paths = list_images(image_dir)
    rng = np.random.default_rng(seed)
    codec = ADJSCCCodec(
        adjscc_weights,
        transmit_channel_num=transmit_channel_num,
        image_size=image_size,
        seed=seed,
        device=tf_device,
    )
    samples: List[Tuple[Path, int]] = [
        (path, crop_index) for path in paths for crop_index in range(crops_per_image)
    ]
    low_batches, high_batches, image_batches, names = [], [], [], []
    for start in range(0, len(samples), batch_size):
        current = samples[start : start + batch_size]
        images = np.stack(
            [load_crop(path, image_size, random_crop, rng) for path, _ in current]
        )
        low = codec.encode(images, low_snr)
        high = codec.encode(images, high_snr)
        # PyTorch側の畳み込みに合わせてNHWCからNCHWへ変換する。
        low_batches.append(torch.from_numpy(np.transpose(low, (0, 3, 1, 2))).half())
        high_batches.append(torch.from_numpy(np.transpose(high, (0, 3, 1, 2))).half())
        # validation時の画像PSNR計算用。crop済みRGBをuint8 NHWCで保持する。
        image_batches.append(torch.from_numpy(np.rint(images).astype(np.uint8)))
        names.extend(f"{path.name}#crop{index}" for path, index in current)
        print(f"信号ペア作成: {min(start + len(current), len(samples))}/{len(samples)}", flush=True)

    low_tensor = torch.cat(low_batches)
    high_tensor = torch.cat(high_batches)
    metadata = {
        "signal_processing": SIGNAL_PROCESSING_VERSION,
        "image_dir": str(Path(image_dir).expanduser().resolve()),
        "adjscc_weights": str(Path(adjscc_weights).expanduser().resolve()),
        "adjscc_fingerprint": checkpoint_fingerprint(adjscc_weights),
        "low_snr": float(low_snr),
        "high_snr": float(high_snr),
        "transmit_channel_num": transmit_channel_num,
        "image_size": image_size,
        "crops_per_image": crops_per_image,
        "random_crop": random_crop,
        "seed": seed,
        "count": len(names),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "low": low_tensor, "high": high_tensor,
        "images": torch.cat(image_batches), "names": names, "metadata": metadata,
    }, destination)
    print(f"信号ペアを保存しました: {destination}")


def load_cache(path: str) -> Dict:
    cache = torch.load(Path(path).expanduser(), map_location="cpu")
    required = {"low", "high", "images", "names", "metadata"}
    if not required.issubset(cache):
        raise RuntimeError(f"不正な信号cacheです（不足: {required - set(cache)}）: {path}")
    return cache


def validate_cache(cache: Dict, expected: Dict) -> None:
    metadata = cache["metadata"]
    mismatches = [f"{key}: cache={metadata.get(key)!r}, expected={value!r}"
                  for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise RuntimeError("信号cacheの条件が一致しません。--rebuild_cacheを指定してください。\n" + "\n".join(mismatches))


class SignalPairDataset(Dataset):
    """raw信号を学習cache共通の範囲で[0,1]へ正規化する。"""

    def __init__(self, cache: Dict, latent_min: float, latent_max: float):
        if latent_max <= latent_min:
            raise ValueError("latent_maxはlatent_minより大きくしてください")
        self.low = cache["low"]
        self.high = cache["high"]
        self.latent_min = latent_min
        self.scale = latent_max - latent_min

    def __len__(self) -> int:
        return len(self.low)

    def __getitem__(self, index: int):
        low = ((self.low[index].float() - self.latent_min) / self.scale).clamp(0.0, 1.0)
        high = ((self.high[index].float() - self.latent_min) / self.scale).clamp(0.0, 1.0)
        return low, high


class AWGNSignalPairDataset(Dataset):
    """独立な通信路AWGNを加え、ADJSCC専用batch dictを返す。"""

    def __init__(
        self, cache: Dict, latent_min: float, latent_max: float,
        channel_snr_min_db: float, channel_snr_max_db: float,
        deterministic_seed: Optional[int] = None,
    ) -> None:
        if latent_max <= latent_min:
            raise ValueError("latent_maxはlatent_minより大きくしてください")
        if channel_snr_min_db < MINIMUM_CHANNEL_SNR_DB:
            raise ValueError(f"channel_snr_min_dbは{MINIMUM_CHANNEL_SNR_DB:g} dB以上にしてください")
        if channel_snr_max_db < channel_snr_min_db:
            raise ValueError("channel_snr_max_dbはchannel_snr_min_db以上にしてください")
        self.low = cache["low"]
        self.high = cache["high"]
        self.images = cache["images"]
        self.latent_min = float(latent_min)
        self.latent_max = float(latent_max)
        self.channel_snr_min_db = float(channel_snr_min_db)
        self.channel_snr_max_db = float(channel_snr_max_db)
        self.deterministic_seed = deterministic_seed

    def __len__(self) -> int:
        return len(self.low)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # raw_s_*はADJSCC Encoderですでに送信電力正規化済み [C,H,W]。
        raw_s_low = self.low[index].float()
        raw_s_high = self.high[index].float()
        generator = None
        if self.deterministic_seed is not None:
            generator = torch.Generator().manual_seed(self.deterministic_seed + index)
        if raw_s_low.ndim != 3 or raw_s_low.shape != raw_s_high.shape:
            raise RuntimeError("cache信号は同一shapeのCHW tensorである必要があります")
        if self.channel_snr_min_db == self.channel_snr_max_db:
            channel_snr_db = raw_s_low.new_tensor(self.channel_snr_min_db)
        else:
            channel_snr_db = torch.empty((), dtype=raw_s_low.dtype).uniform_(
                self.channel_snr_min_db, self.channel_snr_max_db, generator=generator
            )
        # 通信路noiseはここだけで生成する。拡散noiseはmodel.training_stepで別途生成する。
        raw_y, _eps_channel = add_channel_awgn(
            raw_s_low, channel_snr_db, generator=generator
        )
        values = (raw_s_low, raw_s_high, raw_y)
        if not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError("ADJSCC信号またはAWGN受信信号にNaN/Infがあります")
        return {
            "s_low": model_domain(raw_s_low, self.latent_min, self.latent_max),
            "y": model_domain(raw_y, self.latent_min, self.latent_max),
            "s_high": model_domain(raw_s_high, self.latent_min, self.latent_max),
            "channel_snr_db": channel_snr_db,
            "image": self.images[index],  # uint8 [H,W,3]
        }


def latent_range(train_cache: Dict) -> Tuple[float, float]:
    minimum = min(float(train_cache["low"].min()), float(train_cache["high"].min()))
    maximum = max(float(train_cache["low"].max()), float(train_cache["high"].max()))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise RuntimeError(f"不正な信号範囲です: min={minimum}, max={maximum}")
    return minimum, maximum
