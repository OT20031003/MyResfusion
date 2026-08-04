"""RaindropのGT画像だけでADJSCCを学習・評価する。"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# TensorFlowをimportする前にC++ backendのログレベルを設定する。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam

from dataset import dataset_raindrop
from util_channel import Channel
from util_module import Attention_Decoder, Attention_Encoder


def configure_runtime(seed: int) -> None:
    """TensorFlowの再現性とPyTorchとのGPUメモリ共存を設定する。"""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            raise RuntimeError(
                "GPU memory growth must be configured before TensorFlow initializes the GPU"
            ) from error
    tf.keras.utils.set_random_seed(seed)


def build_model(
    image_shape: Tuple[Optional[int], Optional[int], int],
    transmit_channel_num: int,
    learning_rate: float,
    channel_type: str,
) -> Model:
    """SNR適応型Encoder、通信路、Decoderを1つのKeras modelにする。"""
    if channel_type != "awgn":
        raise ValueError("adjscc_raindrop.py currently supports only --channel_type awgn")
    if transmit_channel_num <= 0 or transmit_channel_num % 2 != 0:
        raise ValueError("transmit_channel_num must be a positive even integer")

    input_images = Input(shape=image_shape, name="gt_image")
    input_snr_db = Input(shape=(1,), name="snr_db")
    normalized = Lambda(lambda image: image / 255.0, name="normalize")(input_images)
    encoded = Attention_Encoder(
        normalized, input_snr_db, transmit_channel_num
    )
    received = Channel(channel_type=channel_type)(encoded, input_snr_db)
    decoded = Attention_Decoder(received, input_snr_db)
    output_images = Lambda(lambda image: image * 255.0, name="denormalize")(decoded)

    model = Model(
        inputs=[input_images, input_snr_db], outputs=output_images, name="adjscc_raindrop"
    )
    def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 0.0, 255.0)
        return tf.image.psnr(y_true, y_pred, max_val=255.0)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[psnr],
    )
    return model


def _format_number(value: float) -> str:
    return f"{value:g}"


def experiment_name(args: argparse.Namespace) -> str:
    """checkpointとlogに共通で使う実験名を作る。"""
    return (
        f"adjscc_raindrop_{args.channel_type}"
        f"_tcn{args.transmit_channel_num}"
        f"_snrdb{_format_number(args.snr_low_train)}"
        f"to{_format_number(args.snr_up_train)}"
        f"_bs{args.batch_size}"
        f"_lr{_format_number(args.learning_rate)}"
    )


def resolve_weights_path(args: argparse.Namespace) -> Path:
    if args.load_model_path:
        return Path(args.load_model_path).expanduser()
    return Path(args.model_dir).expanduser() / f"{experiment_name(args)}.h5"


def validate_weights_file(weights_path: Path) -> None:
    """TFCの全重みを含む従来形式のHDF5かを確認する。"""
    with h5py.File(weights_path, "r") as file:
        if "_layer_checkpoint_dependencies" in file:
            raise RuntimeError(
                f"Incompatible checkpoint format: {weights_path}\n"
                "The '.weights.h5' Keras format omits TensorFlow Compression "
                "SignalConv/GDN parameters in this project. Retrain and save "
                "to the legacy '.h5' format."
            )
        if "en1_conv" not in file or "de5_conv" not in file:
            raise RuntimeError(
                f"SignalConv weights were not found in checkpoint: {weights_path}"
            )


def train(args: argparse.Namespace, model: Model) -> None:
    """train/gtを入力と教師の両方に使って学習する。"""
    model_dir = Path(args.model_dir).expanduser()
    loss_dir = Path(args.loss_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_count = dataset_raindrop.get_train_dataset(
        root_dir=args.data_dir,
        snr_low=args.snr_low_train,
        snr_high=args.snr_up_train,
        seed=args.seed,
    )
    if args.batch_size > train_count:
        raise ValueError(
            f"batch_size ({args.batch_size}) exceeds the number of GT images ({train_count})"
        )

    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    steps_per_epoch = train_count // args.batch_size

    validation_dir = args.val_dir or str(Path(args.data_dir) / "val")
    validation_dataset, validation_count = dataset_raindrop.get_validation_dataset(
        validation_dir=validation_dir,
        snr_db=args.val_snr_db,
    )
    validation_dataset = validation_dataset.batch(
        args.val_batch_size, drop_remainder=False
    )
    validation_dataset = validation_dataset.prefetch(
        tf.data.experimental.AUTOTUNE
    )

    # .weights.h5ではTFC固有重みが欠落するため、従来形式の.h5を使う。
    weights_path = model_dir / f"{experiment_name(args)}.h5"
    if args.load_model_path:
        load_path = Path(args.load_model_path).expanduser()
        if not load_path.is_file():
            raise FileNotFoundError(f"Checkpoint was not found: {load_path}")
        validate_weights_file(load_path)
        model.load_weights(str(load_path))
        print(f"Loaded weights: {load_path}")

    checkpoint = ModelCheckpoint(
        filepath=str(weights_path),
        monitor="val_psnr",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    csv_logger = CSVLogger(
        filename=str(loss_dir / f"{experiment_name(args)}.csv"),
        append=bool(args.load_model_path),
    )

    print(f"Training GT images: {train_count}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Training SNR: [{args.snr_low_train}, {args.snr_up_train}) dB")
    print(f"Validation directory: {Path(validation_dir).expanduser()}")
    print(f"Validation images: {validation_count}")
    print(f"Validation SNR: {args.val_snr_db} dB")
    print(f"Checkpoint: {weights_path}")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        callbacks=[checkpoint, csv_logger],
    )

    history_path = loss_dir / f"{experiment_name(args)}.json"
    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_history, file, ensure_ascii=False, indent=2)
    print(f"Training history: {history_path}")


def _evaluate_once(
    model: Model,
    image_paths: List[str],
    snr_db: float,
    save_dir: Optional[Path] = None,
) -> Tuple[float, float]:
    image_mses = []
    image_psnrs = []
    snr_tensor = tf.constant([[snr_db]], dtype=tf.float32)

    for image_path in image_paths:
        target = dataset_raindrop.load_eval_image(image_path)
        target_batch = target[tf.newaxis, ...]
        prediction = model([target_batch, snr_tensor], training=False)
        prediction = tf.clip_by_value(prediction, 0.0, 255.0)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            output_image = tf.cast(tf.round(prediction[0]), tf.uint8)
            encoded_png = tf.image.encode_png(output_image)
            output_path = save_dir / Path(image_path).with_suffix(".png").name
            tf.io.write_file(str(output_path), encoded_png)

        mse = tf.reduce_mean(tf.math.squared_difference(prediction, target_batch))
        psnr = tf.image.psnr(prediction, target_batch, max_val=255.0)
        image_mses.append(float(mse.numpy()))
        image_psnrs.append(float(psnr[0].numpy()))

    return float(np.mean(image_mses)), float(np.mean(image_psnrs))


def evaluate(args: argparse.Namespace, model: Model) -> None:
    """test_a/gtを使い、各整数SNRでMSEとPSNRを測定する。"""
    if args.snr_low_eval > args.snr_up_eval:
        raise ValueError("snr_low_eval must be less than or equal to snr_up_eval")
    if args.eval_repeats <= 0:
        raise ValueError("eval_repeats must be positive")

    weights_path = resolve_weights_path(args)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {weights_path}. "
            "Specify it with --load_model_path."
        )
    validate_weights_file(weights_path)
    model.load_weights(str(weights_path))
    print(f"Loaded weights: {weights_path}")

    image_paths = dataset_raindrop.get_gt_paths(args.data_dir, "test_a")
    image_root = Path(args.eval_dir).expanduser() / "reconstructed" / weights_path.stem
    results: List[Dict[str, float]] = []
    for snr_db in range(args.snr_low_eval, args.snr_up_eval + 1):
        repeated_mses = []
        repeated_psnrs = []
        for repeat_index in range(args.eval_repeats):
            # 評価を複数回繰り返す場合も、PNGは各SNRの1回目だけ保存する。
            save_dir = None
            if args.save_images and repeat_index == 0:
                save_dir = image_root / f"snr_{snr_db:+d}dB"
            mse, psnr = _evaluate_once(
                model, image_paths, float(snr_db), save_dir=save_dir
            )
            repeated_mses.append(mse)
            repeated_psnrs.append(psnr)

        result = {
            "snr_db": snr_db,
            "mse_mean": float(np.mean(repeated_mses)),
            "mse_std": float(np.std(repeated_mses)),
            "psnr_mean": float(np.mean(repeated_psnrs)),
            "psnr_std": float(np.std(repeated_psnrs)),
        }
        results.append(result)
        print(
            f"SNR {snr_db:>3} dB | "
            f"MSE {result['mse_mean']:.6f} +/- {result['mse_std']:.6f} | "
            f"PSNR {result['psnr_mean']:.4f} +/- {result['psnr_std']:.4f} dB"
        )

    eval_dir = Path(args.eval_dir).expanduser()
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / (
        f"{experiment_name(args)}"
        f"_eval{args.snr_low_eval}to{args.snr_up_eval}.json"
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "weights": str(weights_path),
                "image_count": len(image_paths),
                "repeats": args.eval_repeats,
                "results": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Evaluation result: {output_path}")
    if args.save_images:
        print(f"Reconstructed PNG images: {image_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate ADJSCC using only Raindrop GT images"
    )
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--data_dir", default="../../datasets/Raindrop")
    parser.add_argument(
        "--val_dir",
        default=None,
        help="directory containing validation images (default: <data_dir>/val)",
    )
    parser.add_argument("-ct", "--channel_type", default="awgn", choices=("awgn",))
    parser.add_argument("-md", "--model_dir", default="model/")
    parser.add_argument("-lmp", "--load_model_path")
    parser.add_argument("-ldd", "--loss_dir", default="loss/")
    parser.add_argument("-ed", "--eval_dir", default="eval/")
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("-e", "--epochs", default=500, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-tcn", "--transmit_channel_num", default=16, type=int)
    parser.add_argument("--snr_low_train", default=-10.0, type=float)
    parser.add_argument("--snr_up_train", default=20.0, type=float)
    parser.add_argument("--snr_low_eval", default=-10, type=int)
    parser.add_argument("--snr_up_eval", default=20, type=int)
    parser.add_argument(
        "--val_snr_db",
        default=10.0,
        type=float,
        help="fixed AWGN SNR used for validation after every epoch",
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--eval_repeats", default=1, type=int)
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="save reconstructed PNG images from the first repeat at each SNR",
    )
    parser.add_argument("--seed", default=2024, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.seed)
    print("Execution parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    # 学習・検証・テストの全てで同じ空間サイズを使う。
    image_shape = (256, 256, 3)
    model = build_model(
        image_shape=image_shape,
        transmit_channel_num=args.transmit_channel_num,
        learning_rate=args.learning_rate,
        channel_type=args.channel_type,
    )
    model.summary()

    if args.command == "train":
        train(args, model)
    else:
        evaluate(args, model)


if __name__ == "__main__":
    main()
