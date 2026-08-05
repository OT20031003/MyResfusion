# AWGN received-y Standard Resfusion

この新規経路では、AWGN受信信号 `y` を劣化入力として扱い、`residual = y - s_high` を学習する。旧noise-completion経路とcheckpoint、cache、log、評価出力を分離している。

## 学習

```bash
python train_adjscc_signal_resfusion_standard.py \
  --data_dir ../datasets/Raindrop \
  --train_gt_dir ../datasets/Raindrop/train/gt \
  --val_dir ../datasets/Raindrop/val \
  --adjscc_weights ADJSCC/model/adjscc_raindrop_awgn_tcn16_snrdb-10to20_bs8_lr0.0001.h5 \
  --low_snr -10.0 \
  --high_snr 20.0 \
  --channel-snr-min-db -4.0 \
  --channel-snr-max-db 20.0 \
  --transmit_channel_num 16 \
  --image_size 256 \
  --cache_dir standard_received_y_cache \
  --crops_per_image 1 \
  --encoder_batch_size 4 \
  --tf_device /GPU:0 \
  --rebuild_cache \
  --epochs 500 \
  --batch_size 4 \
  --num_workers 4 \
  --pin_mem \
  --check_val_every_n_epoch 1 \
  --accum_iter 1 \
  --gradient_clip 1.0 \
  --precision 32 \
  --seed 2024 \
  --noise_schedule LinearPro \
  --T 12 \
  --mode epsilon \
  --loss_type L2 \
  --optimizer_type AdamW \
  --lr_scheduler_type CosineAnnealingLR \
  --dim 64 \
  --resnet_block_groups 8 \
  --blr 8.8e-4 \
  --min_lr 3e-5 \
  --weight_decay 0.0 \
  --accelerator gpu \
  --devices 1 \
  --num_nodes 1 \
  --log_dir standard_received_y_train \
  --matmul_precision high \
  --checkpoint-monitor val_image_PSNR
```

## 評価

```bash
python test_awgn_adjscc_signal_resfusion_standard.py \
  --resfusion_ckpt standard_received_y_train/lightning_logs/version_0/checkpoints/last.ckpt \
  --channel_snr_db -4.0 \
  --adjscc_weights ADJSCC/model/adjscc_raindrop_awgn_tcn16_snrdb-10to20_bs8_lr0.0001.h5 \
  --input_dir ../datasets/Raindrop/test_a/gt \
  --output_dir standard_received_y_eval \
  --low_snr -10.0 \
  --high_snr 20.0 \
  --device cuda \
  --tf_device /GPU:0 \
  --metrics_device cpu \
  --metrics_batch_size 8 \
  --seed 2024 \
  --limit 100 \
  --sampling-mode stochastic \
  --output-power-normalize
```

決定論的逆拡散は `--sampling-mode deterministic`、出力latentの電力正規化を無効化する比較は `--no-output-power-normalize` を指定する。全画像評価では `--limit` を外す。

## smoke test

```bash
pytest -q test_adjscc_standard_received_y.py
```
