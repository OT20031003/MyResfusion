# received-y Standard Resfusion

## 実装時の独自解釈

- 新方式は既存実装を上書きせず、ファイル名に `_standard` を付けた新規entrypoint・dataset・modelとして分離した。旧noise-completion方式は比較実験用として残しているが、新方式からは呼び出さない。
- Resfusionが扱う値は固定cache統計によるmodel-domainとし、`s_low`、`s_high`、AWGN受信信号`y`へ同じ変換を適用する。AWGN後の`y`は範囲外も保持し、clamp、再電力正規化、sample別rescaleを行わない。
- 標準推論初期化も学習と同じmodel-domainで行う。物理信号上で初期化してからaffine変換するとoffsetが余分に入るためである。
- 通信路AWGNは複素symbol電力`P=1`に対するSNRと解釈した。`channel_epsilon`は円対称複素Gaussianとして、実部・虚部に対応する各実数成分の分散を`1/2`とした。
- 学習時の`diffusion_epsilon`は仕様どおり`torch.randn_like(s_high)`で生成する。通信路noiseとは別の乱数呼び出しであり、分散の比較、差し引き、noise共有は行わない。
- validationではcheckpoint比較を安定させるため、sample indexからchannel SNRとAWGNを固定し、batch indexからdiffusion/posterior noiseも固定した。trainingのSNRとAWGNは毎回再生成する。
- validation image PSNR/LPIPSでは、Resfusion出力だけをADJSCC Decoder投入前に送信電力正規化する。入力`y`、開始state、逆拡散途中stateは正規化しない。
- testでは出力latentの電力正規化をアブレーション可能にした。既存Decoderの前提に合わせ、デフォルトは有効とした。
- checkpoint monitorはデフォルトを`val_image_PSNR`としたが、`val_latent_PSNR`、`val_latent_MSE`、`val_loss`もCLIから選択可能にした。
- 新checkpointはネットワークshapeが旧方式と同じでも残差教師信号が異なるため、旧checkpointとの意味的互換性はない。標準評価scriptはmetadata不一致をValueErrorにする。
- testの標準経路には`additional_variance`、`additional_std`、channel-noise completionを導入しない。旧方式は既存の`test_awgn_adjscc_signal_resfusion.py`にのみ残す。

## 学習データフロー

```text
image
  +-- Encoder(low_encoder_snr)  -> power normalize -> s_low
  |                                                    |
  |                                                    +-- channel AWGN -> y
  +-- Encoder(high_encoder_snr) -> power normalize -> s_high

residual = y - s_high
diffusion_epsilon = independent N(0, I)
t ~ Uniform({0, ..., T_acc - 1})

x_t = sqrt(alpha_bar_t) * s_high
    + (1 - sqrt(alpha_bar_t)) * (y - s_high)
    + sqrt(1 - alpha_bar_t) * diffusion_epsilon

pred_resnoise = U-Net(x_t, y, channel_snr_db, t)
loss = MSE(pred_resnoise, target_resnoise)
```

## 新規学習コマンド

リポジトリルート`/mnt/d/WSL_Work/Resfusion`から実行する。非公開のcache subprocess引数と、新規開始時には値が存在しない`--resume_ckpt`を除く全ユーザー向け引数を明示している。

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
  --encoder_batch_size 8 \
  --tf_device /GPU:0 \
  --rebuild_cache \
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
  --weight_decay 0.0 \
  --accelerator gpu \
  --devices 1 \
  --num_nodes 1 \
  --log_dir standard_received_y_train \
  --matmul_precision high \
  --checkpoint-monitor val_image_PSNR
```

初回またはcache形式更新後は`--rebuild_cache`を指定する。再利用時は外す。TensorFlow Encoder/DecoderをCPUで実行する場合は`--tf_device /CPU:0`、pin memoryを無効化する場合は`--pin_mem`を`--no_pin_mem`へ置き換える。

### 学習再開

上のコマンドから`--rebuild_cache`を外し、実際のcheckpointパスを追加する。

```bash
  --resume_ckpt standard_received_y_train/lightning_logs/version_0/checkpoints/last.ckpt
```

## 標準AWGN評価コマンド

評価parserの全引数を明示している。`--limit 100`を外すと全画像を評価する。

```bash
python test_awgn_adjscc_signal_resfusion_standard.py \
  --resfusion_ckpt standard_received_y_train/lightning_logs/version_1/checkpoints/last.ckpt \
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

決定論的逆拡散では次を置き換える。

```bash
--sampling-mode deterministic
```

出力latentを電力正規化しない比較では次を置き換える。

```bash
--no-output-power-normalize
```

## Smoke test

pytestが導入済みの場合:

```bash
pytest -q test_adjscc_standard_received_y.py
```

現在の`resfusion-adjscc`環境にはpytestがなかったため、実装確認時は次の直接実行と同等の方法で全test関数を実行した。

```bash
MPLCONFIGDIR=/tmp/matplotlib-resfusion \
PYTHONDONTWRITEBYTECODE=1 \
/home/naaa/miniconda3/envs/resfusion-adjscc/bin/python - <<'PY'
import test_adjscc_standard_received_y as tests

names = [name for name in dir(tests) if name.startswith("test_")]
for name in names:
    getattr(tests, name)()
    print("PASS", name)
print("TOTAL", len(names))
PY
```

確認結果:

```text
PASS test_condition_is_y_and_backward_is_finite
PASS test_dataset_rejects_below_minus_four_and_keeps_out_of_range_y
PASS test_model_domain_roundtrip_has_no_clamp
PASS test_received_y_residual_forward_formula_and_independent_noise
PASS test_standard_initialization_exact_formula_without_completion_variance
TOTAL 5
```

## 新旧entrypoint

```text
旧学習:
  train_adjscc_signal_resfusion.py

新しいreceived-y標準学習:
  train_adjscc_signal_resfusion_standard.py

旧noise-completion評価:
  test_awgn_adjscc_signal_resfusion.py

新しい標準評価:
  test_awgn_adjscc_signal_resfusion_standard.py
```

新方式のcache、log、評価出力は、それぞれ`standard_received_y_cache`、`standard_received_y_train`、`standard_received_y_eval`をデフォルトとする。
