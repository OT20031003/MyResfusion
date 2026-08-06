"""256x256のFFHQ PNGでADJSCCを学習する。"""

import argparse
import json
import random
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from adjscc_raindrop import build_model, configure_runtime, validate_weights_file


def experiment_name(args: argparse.Namespace) -> str:
    return (
        f"adjscc_ffhq_{args.channel_type}"
        f"_tcn{args.transmit_channel_num}"
        f"_snrdb{args.snr_low_train:g}to{args.snr_up_train:g}"
        f"_bs{args.batch_size}_lr{args.learning_rate:g}"
    )


def load_sample(path: tf.Tensor, snr_low: tf.Tensor, snr_high: tf.Tensor):
    """PNGを変形せず読み、サンプルごとの学習SNRを付与する。"""
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.ensure_shape(image, [256, 256, 3])
    image = tf.cast(image, tf.float32)
    snr_db = tf.random.uniform(
        [1], minval=snr_low, maxval=snr_high, dtype=tf.float32
    )
    return (image, snr_db), image


def load_validation_sample(path: tf.Tensor, snr_db: tf.Tensor):
    """検証PNGを変形せず読み、固定SNRを付与する。"""
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.ensure_shape(image, [256, 256, 3])
    image = tf.cast(image, tf.float32)
    return (image, snr_db), image


def build_datasets(args: argparse.Namespace):
    data_dir = Path(args.data_dir).expanduser()
    paths = sorted(str(path) for path in data_dir.glob("*.png") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"PNG images were not found: {data_dir}/*.png")
    if args.snr_low_train >= args.snr_up_train:
        raise ValueError("--snr_low_train must be smaller than --snr_up_train")
    if args.val_count <= 0 or args.val_count >= len(paths):
        raise ValueError("--val_count must be between 1 and image_count - 1")

    # ファイル名への依存を避けつつ、再実行時には同じhold-outになる固定分割。
    random.Random(args.seed).shuffle(paths)
    validation_paths = paths[:args.val_count]
    training_paths = paths[args.val_count:]

    snr_low = tf.constant(args.snr_low_train, dtype=tf.float32)
    snr_high = tf.constant(args.snr_up_train, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(training_paths)
    train_dataset = train_dataset.shuffle(
        len(training_paths), seed=args.seed, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.map(
        lambda path: load_sample(path, snr_low, snr_high),
        num_parallel_calls=args.num_parallel_calls,
    )
    # drop_remainder=Falseなので、各epochですべての画像をちょうど1回使用する。
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=False)

    val_snr = tf.constant([args.val_snr_db], dtype=tf.float32)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_paths)
    validation_dataset = validation_dataset.map(
        lambda path: load_validation_sample(path, val_snr),
        num_parallel_calls=args.num_parallel_calls,
    )
    validation_dataset = validation_dataset.batch(
        args.val_batch_size, drop_remainder=False
    )
    return (
        train_dataset.prefetch(args.prefetch_batches),
        # 画像と教師をcacheすると約2 GBのpinned host memoryを要求するため、
        # validation PNGもbatchごとに逐次読み込む。
        validation_dataset.prefetch(args.prefetch_batches),
        len(training_paths),
        len(validation_paths),
    )


def train(args: argparse.Namespace) -> None:
    train_dataset, validation_dataset, train_count, val_count = build_datasets(args)
    model = build_model(
        image_shape=(256, 256, 3),
        transmit_channel_num=args.transmit_channel_num,
        learning_rate=args.learning_rate,
        channel_type=args.channel_type,
    )

    model_dir = Path(args.model_dir).expanduser()
    loss_dir = Path(args.loss_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)
    name = experiment_name(args)
    weights_path = model_dir / f"{name}.h5"

    if args.load_model_path:
        load_path = Path(args.load_model_path).expanduser()
        if not load_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not found: {load_path}")
        validate_weights_file(load_path)
        model.load_weights(str(load_path))
        print(f"Loaded weights: {load_path}")

    checkpoint = ModelCheckpoint(
        filepath=str(weights_path), monitor="val_psnr", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1,
    )
    csv_path = loss_dir / f"{name}.csv"
    csv_logger = CSVLogger(str(csv_path), append=bool(args.load_model_path))

    print(f"FFHQ training images: {train_count}")
    print(f"FFHQ validation images: {val_count}")
    print(f"Batches per epoch: {(train_count + args.batch_size - 1) // args.batch_size}")
    print("Image processing: decode PNG only (no resize, crop, or augmentation)")
    print(f"Training SNR: [{args.snr_low_train}, {args.snr_up_train}) dB")
    print(f"Validation SNR: {args.val_snr_db} dB")
    print(f"Checkpoint: {weights_path}")
    history = model.fit(
        train_dataset, epochs=args.epochs, validation_data=validation_dataset,
        callbacks=[checkpoint, csv_logger],
    )

    history_path = loss_dir / f"{name}.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {key: [float(value) for value in values]
             for key, values in history.history.items()},
            file, ensure_ascii=False, indent=2,
        )
    print(f"Training history: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../datasets/ffhq_train_70k")
    parser.add_argument("--channel_type", default="awgn", choices=("awgn",))
    parser.add_argument("--model_dir", default="ADJSCC/model/ffhq")
    parser.add_argument("--loss_dir", default="ADJSCC/loss/ffhq")
    parser.add_argument("--load_model_path")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--val_batch_size", default=8, type=int)
    parser.add_argument("--val_count", default=1000, type=int)
    parser.add_argument("--val_snr_db", default=10.0, type=float)
    parser.add_argument("--num_parallel_calls", default=4, type=int)
    parser.add_argument("--prefetch_batches", default=1, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--transmit_channel_num", default=16, type=int)
    parser.add_argument("--snr_low_train", default=-10.0, type=float)
    parser.add_argument("--snr_up_train", default=20.0, type=float)
    parser.add_argument("--seed", default=2024, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configure_runtime(arguments.seed)
    train(arguments)
