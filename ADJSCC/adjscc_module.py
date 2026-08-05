"""学習済みADJSCCをチャネルなしのEncoder/Decoderとして利用する。"""

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

try:
    from .util_module import Attention_Decoder, Attention_Encoder
except ImportError:  # ADJSCCディレクトリから直接実行する場合
    from util_module import Attention_Decoder, Attention_Encoder


def configure_tensorflow(seed: int = 2024) -> None:
    """再現性を設定し、TensorFlowによるGPUメモリの全量確保を防ぐ。"""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.keras.utils.set_random_seed(seed)


def _validate_legacy_weights(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"ADJSCC checkpointが見つかりません: {path}")
    with h5py.File(path, "r") as file:
        if "_layer_checkpoint_dependencies" in file:
            raise RuntimeError(
                "この.weights.h5はTFCのSignalConv/GDN重みが欠落する形式です。"
                " adjscc_raindrop.pyで保存した従来形式の.h5を指定してください。"
            )
        if "en1_conv" not in file or "de5_conv" not in file:
            raise RuntimeError(f"Encoder/Decoder重みを確認できません: {path}")


class ADJSCCCodec:
    """凍結済みADJSCCのチャネル直前Encoderとチャネル直後Decoder。

    入出力画像は0～255、信号はNHWC形式である。電力正規化のみ適用し、
    通信路ノイズは一切適用しない。
    """

    def __init__(
        self,
        weights_path: str,
        transmit_channel_num: int = 16,
        image_size: int = 256,
        seed: int = 2024,
        device: Optional[str] = None,
    ) -> None:
        if transmit_channel_num <= 0 or transmit_channel_num % 2:
            raise ValueError("transmit_channel_numは正の偶数にしてください")
        configure_tensorflow(seed)
        self.image_size = image_size
        self.transmit_channel_num = transmit_channel_num
        weights = Path(weights_path).expanduser().resolve()
        _validate_legacy_weights(weights)

        scope = tf.device(device) if device else nullcontext()
        with scope:
            image = Input((image_size, image_size, 3), name="gt_image")
            snr = Input((1,), name="snr_db")
            normalized = Lambda(lambda value: value / 255.0, name="normalize")(image)
            signal = Attention_Encoder(normalized, snr, transmit_channel_num)
            self.encoder = Model([image, snr], signal, name="adjscc_encoder")

            latent = Input(
                (image_size // 4, image_size // 4, transmit_channel_num),
                name="encoded_signal",
            )
            decoder_snr = Input((1,), name="decoder_snr_db")
            decoded = Attention_Decoder(latent, decoder_snr)
            output = Lambda(lambda value: value * 255.0, name="denormalize")(decoded)
            self.decoder = Model([latent, decoder_snr], output, name="adjscc_decoder")

        # 元の一体型modelと同名の層を、従来HDF5から名前で復元する。
        self.encoder.load_weights(str(weights), by_name=True, skip_mismatch=False)
        self.decoder.load_weights(str(weights), by_name=True, skip_mismatch=False)
        self.encoder.trainable = False
        self.decoder.trainable = False

    @staticmethod
    def _snr(batch_size: int, snr_db: float) -> tf.Tensor:
        return tf.fill((batch_size, 1), tf.cast(snr_db, tf.float32))

    @staticmethod
    def power_normalize(signals: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """Channel層と同じ平均複素シンボル電力1の制約だけを適用する。

        NHWC信号全体をflattenし、前半を実部、後半を虚部として扱う。
        AWGNは追加しない。
        """
        signals = np.asarray(signals, dtype=np.float32)
        original_shape = signals.shape
        flattened = signals.reshape(signals.shape[0], -1)
        if flattened.shape[1] % 2:
            raise ValueError("複素信号化するため、信号要素数は偶数である必要があります")
        dim_z = flattened.shape[1] // 2
        complex_signal = flattened[:, :dim_z] + 1j * flattened[:, dim_z:]
        power = np.sum(np.abs(complex_signal) ** 2, axis=1, keepdims=True)
        factor = np.sqrt(dim_z / np.maximum(power, epsilon))
        normalized = complex_signal * factor
        result = np.concatenate([normalized.real, normalized.imag], axis=1)
        return result.reshape(original_shape).astype(np.float32)

    def encode(self, images: np.ndarray, snr_db: float) -> np.ndarray:
        """画像を指定SNR条件でencodeし、電力正規化する（AWGNなし）。"""
        images = np.asarray(images, dtype=np.float32)
        result = self.encoder([images, self._snr(len(images), snr_db)], training=False)
        return self.power_normalize(result.numpy())

    def decode(self, signals: np.ndarray, snr_db: float) -> np.ndarray:
        """NHWC信号を、指定SNR条件で0～255画像へ復号する。"""
        signals = np.asarray(signals, dtype=np.float32)
        result = self.decoder([signals, self._snr(len(signals), snr_db)], training=False)
        return np.clip(result.numpy(), 0.0, 255.0)
