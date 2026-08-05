# ADJSCC信号空間で動作するMyResfusion

## 実装した処理

目的の推論経路は次のとおりです。

```text
GT image (256x256 RGB)
  -> frozen ADJSCC Encoder (low SNR condition)
  -> power normalization (average complex-symbol power = 1)
  -> low-SNR latent signal [16, 64, 64]
  -> Resfusion
  -> predicted high-SNR latent signal [16, 64, 64]
  -> power normalization (average complex-symbol power = 1)
  -> frozen ADJSCC Decoder (high SNR condition)
  -> reconstructed image (256x256 RGB)
```

通信路ノイズ（AWGNの付加）は、学習時にもテスト時にも使用しません。ただし、元のADJSCC Decoderが前提としている平均複素シンボル電力1の正規化は残します。low/high SNRは物理的な通信路のSNRではなく、ADJSCCのAF（Attention Feature）moduleへ渡す条件値だけを意味します。

追加したファイルは次のとおりです。

- `ADJSCC/adjscc_module.py`: 学習済みADJSCCを `encode(images, snr)` と `decode(signals, snr)` に分離する凍結module
- `adjscc_signal_data.py`: 同じGT cropからlow/high信号ペアを作り、cacheへ保存する処理
- `model/latent_resfusion.py`: 16チャネル信号を検証できるResfusion Lightning module
- `train_adjscc_signal_resfusion.py`: 信号ペアの準備とResfusion学習
- `test_adjscc_signal_resfusion.py`: 指定された推論経路による画像再構成とPNG保存

## 実装時に解釈した点

指示に明記されていなかったため、次のように解釈しました。

1. low SNRは既定値 `-10 dB`、high SNRは既定値 `20 dB` の固定ペアです。各画像でランダムにSNRを変える実装ではありません。引数で変更できます。
2. low/high信号は、同じADJSCC checkpoint、同じGT画像、同じcropから生成します。異なるのはEncoderへ渡すSNR条件だけです。両方に元の`Channel`層と同じ電力正規化を適用しますが、AWGNは適用しません。
3. Resfusionの教師はhigh-SNR Encoder出力です。復号画像に対するlossは使用せず、ADJSCCには勾配を流しません。
4. ADJSCC Encoder/Decoderは凍結します。TensorFlowとPyTorchをまたぐend-to-end学習ではありません。
5. 元のResfusionは入力範囲 `[0,1]` を前提とするため、学習用low/high信号全体に共通する最小値・最大値で信号を `[0,1]` へ変換します。この範囲はResfusion checkpointに保存され、テスト時にも同じ値を使います。学習範囲外のテスト信号はclipされます。
6. 学習画像はRaindropの従来処理に合わせて256角のrandom cropと左右反転を行います。検証・テスト画像は256角のcenter cropです。小さい画像は縦横比を保ったまま拡大してからcropします。
7. 信号ペアは事前計算して再利用します。既定では元画像1枚につき1 cropです。epochごとに別cropへはなりません。増やす場合は `--crops_per_image` を指定してcacheを再作成します。
8. `transmit_channel_num=16` の場合、256角画像から得る信号形状はNHWCで `64x64x16`、Resfusion内ではNCHWで `16x64x64` です。
9. Resfusionは既存の `RDDM_Unet`、`LinearProScheduler(T=12)`、epsilon予測を既定値として使います。
10. テストのResfusion生成には拡散noiseがあるため、seedを変えると結果も変わります。既定のseedは2024です。

## 前提

ADJSCCを再学習して得た従来HDF5形式の `.h5` checkpointが必要です。以前生成された `.weights.h5` はTensorFlow CompressionのSignalConv/GDN重みが欠落するため使用できません。

統合環境で `pkg_resources` が見つからない場合は、先に次を実行してください。`setuptools 81`以降では`pkg_resources`が削除されており、このrepositoryの旧PyTorch Lightningとは互換性がありません。

```bash
conda activate resfusion-adjscc
python -m pip install "setuptools==70.3.0" einops
```

`einops`はMask版だけでなく、今回使う`RDDM_Unet`の内部でも必要です。

## 学習

リポジトリrootで実行します。checkpoint名は手元のものに合わせてください。

```bash
conda activate resfusion-adjscc
cd /mnt/d/WSL_Work/Resfusion

python train_adjscc_signal_resfusion.py \
  --data_dir ../datasets/Raindrop \
  --val_dir ../datasets/Raindrop/val \
  --adjscc_weights ADJSCC/model/adjscc_raindrop_awgn_tcn16_snrdb-10to20_bs8_lr0.0001.h5 \
  --low_snr -10 \
  --high_snr 20 \
  --transmit_channel_num 16 \
  --batch_size 8 \
  --epochs 500 \
  --devices 1 \
  --rebuild_cache
```

初回は次のcacheを自動作成してから、TensorFlow processを終了し、PyTorchの学習を開始します。

```text
my_resfusion_cache/train_signal_pairs.pt
my_resfusion_cache/val_signal_pairs.pt
```

これによりTensorFlowとPyTorchが学習中に同時にGPUメモリを保持することを避けます。GPU版TensorFlowを使えない場合は `--tf_device /CPU:0` を指定できます。cache作成は遅くなりますが、作成後のResfusion学習速度には影響しません。

画像やADJSCC checkpoint、SNR、crop数、信号処理方式などを変更した場合はcacheの不一致を検出して停止します。今回の電力正規化を反映するため、以前のcacheは必ず作り直してください。

```bash
--rebuild_cache
```

学習cropを1画像あたり4個に増やす例です。

```bash
--crops_per_image 4 --rebuild_cache
```

checkpointは通常、次の配下に保存されます。

```text
my_resfusion_train/lightning_logs/version_*/checkpoints/
  best-....ckpt
  last.ckpt
```

`val_latent_PSNR` は正規化したhigh-SNR信号と予測信号のPSNRです。最良checkpointの選択にはこの値を使います。これは復号画像のPSNRではありません。

途中から再開する場合は同じ引数に次を追加します。

```bash
--resume_ckpt my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt
```

## テストとPNG保存

```bash
python test_adjscc_signal_resfusion.py \
  --resfusion_ckpt my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt \
  --input_dir ../datasets/Raindrop/test_a/gt \
  --output_dir my_resfusion_eval
```

ADJSCC checkpointのパス、low/high SNR、信号チャネル数、正規化範囲はResfusion checkpointから読みます。ADJSCC checkpointを移動した場合だけ、次を追加して新しい場所を指定します。

```bash
--adjscc_weights ADJSCC/model/使用するcheckpoint.h5
```

出力は次のとおりです。

```text
my_resfusion_eval/resfusion_high_decoder/*_reconstructed.png
my_resfusion_eval/no_resfusion_low_decoder/*_reconstructed.png
my_resfusion_eval/no_resfusion_high_decoder/*_reconstructed.png
my_resfusion_eval/metrics.csv
```

3ディレクトリはそれぞれ、Resfusion出力をhigh条件で復号、low信号をlow条件で直接復号、low信号をhigh条件で直接復号した結果です。`metrics.csv`には3経路それぞれについて、center cropした入力GTとのPSNRを保存します。動作確認で先頭3枚だけ処理する場合は `--limit 3` を指定します。

## 注意点

- ADJSCC自体がlow/high SNR条件で十分に異なる信号表現を作れていることが前提です。
- channelを外しているため、この実験だけではAWGNに対する通信耐性を評価していません。
- 高SNR Encoder信号へ近づくことと、最終画像PSNRが必ず上がることは同義ではありません。必要なら次の段階で、凍結DecoderをPyTorchへ移植するなどして画像lossも加える設計が必要です。
- cacheはfloat16で保存し、DataLoaderでfloat32へ戻します。これはcache容量を約半分にするための判断です。
