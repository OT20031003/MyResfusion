"""RaindropのGT画像だけを読み込むTensorFlowデータローダ。"""

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def get_gt_paths(root_dir: str, subset: str) -> List[str]:
    """subset/gt配下の画像パスをファイル名順で返す。"""
    gt_dir = Path(root_dir).expanduser() / subset / "gt"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory was not found: {gt_dir}")

    paths = sorted(
        str(path)
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"GT images were not found: {gt_dir}")
    return paths


def get_validation_paths(validation_dir: str) -> List[str]:
    """指定ディレクトリ直下の検証画像パスをファイル名順で返す。"""
    validation_dir = Path(validation_dir).expanduser()
    if not validation_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory was not found: {validation_dir}"
        )

    paths = sorted(
        str(path)
        for path in validation_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"Validation images were not found: {validation_dir}"
        )
    return paths


def read_and_resize(path: tf.Tensor) -> tf.Tensor:
    """Resfusionと同じサイズ規則でRGB画像を読み込む。"""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    long_side = tf.maximum(height, width)

    # 長辺を最大1024にし、その後に縦横を16の倍数へ切り上げる。
    scale = tf.minimum(1.0, 1024.0 / tf.cast(long_side, tf.float32))
    new_height = tf.cast(
        tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32
    )
    new_width = tf.cast(
        tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32
    )
    new_height = 16 * tf.cast(
        tf.math.ceil(tf.cast(new_height, tf.float32) / 16.0), tf.int32
    )
    new_width = 16 * tf.cast(
        tf.math.ceil(tf.cast(new_width, tf.float32) / 16.0), tf.int32
    )

    # Raindrop.pyのPIL resizeデフォルトに合わせて最近傍法を使う。
    return tf.image.resize(
        image,
        [new_height, new_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        antialias=False,
    )


def _make_train_sample(
    path: tf.Tensor,
    snr_low: tf.Tensor,
    snr_high: tf.Tensor,
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    image = read_and_resize(path)
    shape = tf.shape(image)
    tf.debugging.assert_greater_equal(
        shape[0], 256, message="GT image height must be at least 256 pixels"
    )
    tf.debugging.assert_greater_equal(
        shape[1], 256, message="GT image width must be at least 256 pixels"
    )

    # Resfusionの学習と同じく256x256 cropと50%の左右反転を適用する。
    image = tf.image.random_crop(image, [256, 256, 3])
    image = tf.image.random_flip_left_right(image)

    # 各サンプルに独立な連続一様分布のSNRを与える。
    snr_db = tf.random.uniform(
        shape=[1], minval=snr_low, maxval=snr_high, dtype=tf.float32
    )
    return (image, snr_db), image


def get_train_dataset(
    root_dir: str,
    snr_low: float = -10.0,
    snr_high: float = 20.0,
    seed: int = 2024,
) -> Tuple[tf.data.Dataset, int]:
    """train/gtから((GT, SNR), GT)形式の学習datasetを作る。"""
    if snr_low >= snr_high:
        raise ValueError("snr_low must be smaller than snr_high")

    paths = get_gt_paths(root_dir, "train")
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(
        buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
    )
    low = tf.constant(snr_low, dtype=tf.float32)
    high = tf.constant(snr_high, dtype=tf.float32)
    dataset = dataset.map(
        lambda path: _make_train_sample(path, low, high),
        num_parallel_calls=AUTOTUNE,
    )

    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)
    return dataset, len(paths)


def _make_validation_sample(
    path: tf.Tensor,
    snr_db: tf.Tensor,
) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    image = center_crop_256(read_and_resize(path), "Validation")
    return (image, snr_db), image


def center_crop_256(image: tf.Tensor, image_role: str = "Input") -> tf.Tensor:
    """画像中央の256x256を切り出し、ADJSCCの学習サイズに合わせる。"""
    shape = tf.shape(image)
    tf.debugging.assert_greater_equal(
        shape[0], 256, message=f"{image_role} image height must be at least 256 pixels"
    )
    tf.debugging.assert_greater_equal(
        shape[1], 256, message=f"{image_role} image width must be at least 256 pixels"
    )

    offset_height = (shape[0] - 256) // 2
    offset_width = (shape[1] - 256) // 2
    return tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, 256, 256
    )


def get_validation_dataset(
    validation_dir: str,
    snr_db: float = 10.0,
) -> Tuple[tf.data.Dataset, int]:
    """指定フォルダから毎エポックのPSNR計算用datasetを作る。"""
    paths = get_validation_paths(validation_dir)
    snr = tf.constant([snr_db], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(
        lambda path: _make_validation_sample(path, snr),
        num_parallel_calls=AUTOTUNE,
    )
    # 画像の読み込みと中央cropは固定なのでcacheする。
    return dataset.cache(), len(paths)


def load_eval_image(path: str) -> tf.Tensor:
    """test_a/gtの1枚を読み、中央256x256をテスト入力にする。"""
    image = read_and_resize(tf.convert_to_tensor(path, dtype=tf.string))
    return center_crop_256(image, "Test")
