# WR.tex 実装メモ

## 実装したもの

- `adjscc_signal_data_wr.py`: ADJSCC Encoder が出力した電力正規化済み raw latent に対し、`n ~ N(0,1)` と `y = z_low + sigma n` を実装した。受信後の再正規化および clip は行わない。
- `model/adjscc_signal_resfusion_wr.py`: WR.tex の `z_t`、合成 noise `epsilon`、`R=z_low-z_high` を含む resnoise 教師信号、L2 学習、および `z_T'=y/2+sqrt(3-sigma^2) epsilon_add/2` からの逆過程を実装した。
- `train_adjscc_signal_resfusion_wr.py`: 学習専用 entrypoint。`t=1,...,T'` を一様 sampling する。
- `infer_adjscc_signal_resfusion_wr.py`: 推論専用 entrypoint。学習処理は含まない。
- `test_adjscc_signal_resfusion_wr.py`: WR.tex の式、残差、条件入力、`alpha_bar_0=1` を検査する。

## WR.tex に明記されていない補足実装

- SNR[dB] から `sigma=10^(-SNR/20)` へ変換する際、送信信号電力を 1 とした。したがって `sigma^2<=3` から最小 SNR は `-10 log10(3) = -4.771213 dB` である。
- 数式の `t=1` は既存 denoiser の index `0` に対応させた。既存 scheduler の先頭 posterior 値を WR の `alpha_bar_0=1` に合わせ、`alpha_bar_{t-1}=1`, `beta_hat_1=0` とした。
- ネットワークには既存実装と同じ SNR conditioning を追加した（WR.tex は `(z_t,y,t)` のみを明記）。複数 SNR の一つのモデルで `sigma` の違いを識別可能にするためである。
- checkpoint 選択、optimizer、validation latent MSE、乱数 seed、勾配 clipping、cache 作成を運用上必要な機能として追加した。
- 推論後、ADJSCC Decoder に入力する直前の電力再正規化を既定で有効にした。これは WR の forward／reverse 過程の外側であり、`--no-output-power-normalize` で無効化できる。
- Oracle baselineとして、high-SNR条件でencodeした信号に提案法と同じチャネル雑音を加え、Resfusionなしでhigh-SNR条件でdecodeする経路を評価に含めた。
- 既存 Resfusion と同じ posterior sampling を実装した。再現的な平均経路が必要な場合は `--deterministic` で各逆 step の posterior noise を 0 にできる。開始点の `epsilon_add` は常に生成する。

## コマンド例

学習:

```bash
python train_adjscc_signal_resfusion_wr.py \
  --adjscc_weights ADJSCC/model/ffhq/adjscc_ffhq_awgn_tcn16_snrdb-10to20_bs16_lr0.0001.h5 \
  --data_dir ../datasets/ffhq_train_70k \
  --train_gt_dir ../datasets/ffhq_train_70k \
  --val_dir ../datasets/ffhq_val \
  --low_snr -10 \
  --high_snr 20 \
  --channel-snr-min-db -4.0 \
  --channel-snr-max-db 20.0 \
  --transmit_channel_num 16 \
  --image_size 256 \
  --cache_dir wr_signal_cache \
  --crops_per_image 1 \
  --encoder_batch_size 8 \
  --tf_device /GPU:0 \
  --epochs 500 \
  --batch_size 8 \
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
  --weight_decay 0 \
  --accelerator gpu \
  --devices 1 \
  --num_nodes 1 \
  --log_dir wr_signal_train \
  --matmul_precision high \
  --rebuild_cache
```

cacheを作り直す場合は `--rebuild_cache`、checkpointから再開する場合は
`--resume_ckpt wr_signal_train/lightning_logs/version_0/checkpoints/last.ckpt` を追加する。

推論:

```bash
python infer_adjscc_signal_resfusion_wr.py \
  --resfusion_ckpt wr_signal_train/lightning_logs/version_0/checkpoints/last.ckpt \
  --channel_snr_db -4 \
  --adjscc_weights ADJSCC/model/adjscc_raindrop_awgn_tcn16_snrdb-10to20_bs8_lr0.0001.h5 \
  --input_dir ../datasets/Raindrop/test_a/gt \
  --output_dir wr_signal_inference \
  --low_snr -10 \
  --high_snr 20 \
  --device cuda:0 \
  --tf_device /GPU:0 \
  --metrics_device cpu \
  --metrics_batch_size 8 \
  --seed 2024 \
  --sampling-mode stochastic \
  --no-output-power-normalize
```

評価枚数を制限する場合は `--limit 100`、posterior noiseを無効にする場合は
`--sampling-mode deterministic`（または `--deterministic`）、Decoder直前の電力正規化を
無効にする場合は `--no-output-power-normalize` に置き換える。

テスト:

```bash
pytest -q test_adjscc_signal_resfusion_wr.py
```
