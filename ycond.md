# 実装時の解釈

- 通信路SNRは複素シンボル電力 `P=1` に対するdB値と解釈した。実数表現の各成分には標準偏差 `10^(-gamma_db/20) / sqrt(2)` の独立Gaussian noiseを加える。これは既存AWGNテストの円対称複素Gaussian定義と一致する。
- model-domain変換は既存checkpointが保持するtrain cacheの `latent_min` / `latent_max` による固定affine変換を継続した。ただしAWGN経路ではclampを一切行わない。
- validationの通常のResfusion初期化は条件 `y` から行う。`test_awgn_adjscc_signal_resfusion.py` のnoise-completion初期化は既存評価方式として維持し、独立noiseで構成する学習分布とは完全には一致しない旨を同ファイルへ明記した。
- SNR条件は指定された第一候補を採用し、学習範囲を用いてdB値を `[-1,1]` へ固定正規化したMLP embeddingをtime embeddingへ加算する。

# 実行コマンド

以下はリポジトリルート `/mnt/d/WSL_Work/Resfusion` から実行する。

## 学習（新規開始）

内部cache構築専用の非公開引数（`--_build_cache` など）と、新規開始時には値を渡せない `--resume_ckpt` を除き、ユーザー向け引数をすべて明示している。`--rebuild_cache` を指定しているため、初回実行時だけでなく既存cacheも再構築する。既存cacheを再利用する場合は、このフラグだけを外す。

best checkpointはvalidation latentをADJSCC Decoderで画像へ戻した後の `val_image_PSNR` が最大のepochから保存する。同じ復元画像についてAlexNetベースの `val_image_LPIPS` も記録するが、checkpoint選択には使用しない。validationのSNRとAWGNはepoch間で比較可能になるようsampleごとに固定している。cacheには画像metric比較用画像が追加されたため、旧形式cacheを使用していた場合は一度 `--rebuild_cache` が必要になる。

```bash
python train_adjscc_signal_resfusion.py \
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
  --cache_dir my_resfusion_cache \
  --crops_per_image 1 \
  --encoder_batch_size 8 \
  --tf_device /GPU:0 \
  --rebuild_cache \
  --epochs 500 \
  --batch_size 16 \
  --num_workers 8 \
  --pin_mem \
  --check_val_every_n_epoch 10 \
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
  --log_dir my_resfusion_train \
  --matmul_precision high
```

CPUでcacheを構築する場合は `--tf_device /CPU:0`、pin memoryを無効化する場合は `--pin_mem` を `--no_pin_mem` に置き換える。

## 学習再開

上の学習コマンドから `--rebuild_cache` を外し、次の引数を追加する。checkpointパスは実際のLightning出力先に合わせる。

```bash
  --resume_ckpt my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt
```

## AWGN評価

評価parserの引数をすべて明示している。`--resfusion_ckpt` は実際に生成されたcheckpointへ置き換える。`--limit 100` は先頭100画像を評価する指定であり、全画像を評価する場合は、値として `None` を渡せないため `--limit` 引数そのものを外す。

```bash
python test_awgn_adjscc_signal_resfusion.py \
  --resfusion_ckpt my_resfusion_train/lightning_logs/version_3/checkpoints/last.ckpt \
  --channel_snr_db -4.0 \
  --adjscc_weights ADJSCC/model/adjscc_raindrop_awgn_tcn16_snrdb-10to20_bs8_lr0.0001.h5 \
  --input_dir ../datasets/Raindrop/test_a/gt \
  --output_dir my_awgn_resfusion_eval \
  --low_snr -10.0 \
  --high_snr 20.0 \
  --device cuda \
  --tf_device /GPU:0 \
  --metrics_device cpu \
  --metrics_batch_size 8 \
  --seed 2024 \
  --limit 100
```

## 追加unit test

```bash
pytest -q test_adjscc_awgn_training.py
```
