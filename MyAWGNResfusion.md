# AWGN Channel-Noise-Aware Resfusionテスト

## 実装した系列

`test_awgn_adjscc_signal_resfusion.py`は、物理AWGNをResfusion開始noiseの一部として利用する推論専用スクリプトである。

```text
image
  -> ADJSCC Encoder(low-SNR condition)
  -> power normalization (P=1)
  -> physical AWGN(channel_snr_db)
  -> received latent y
  -> noise completion
  -> initial state u_T'
  -> Resfusion reverse process conditioned by y
  -> power normalization
  -> ADJSCC Decoder(high-SNR condition)
  -> reconstructed image
```

AWGN後からnoise completionまで、clipおよびサンプルごとの再電力正規化は行わない。

同じ評価画像について、次の4経路を比較する。

```text
1. 提案法:
   Encoder(low) -> AWGN(gamma_ch) -> noise completion
   -> Resfusion -> Decoder(high)

2. No-Resfusion low baseline:
   Encoder(low) -> Decoder(low)

3. No-Resfusion high baseline:
   Encoder(low) -> Decoder(high)

4. Actual-channel ADJSCC baseline:
   Encoder(gamma_ch) -> AWGN(gamma_ch) -> Decoder(gamma_ch)
```

2と3は先のチャネルなしテストと同じベースラインである。4は通常のSNR適応ADJSCC通信経路に相当する。

## 実装式

平均複素シンボル電力を`P=1`、開始点を`alpha_bar=1/4`と固定する。受信信号は、

```math
y=u_l+n_{ch},\qquad n_{ch}\sim\mathcal{CN}(0,10^{-\gamma_{ch}/10})
```

である。追加noiseの複素標準偏差を、

```math
\lambda_{add}=\frac{1}{2}\sqrt{3-10^{-\gamma_{ch}/10}}
```

とし、逆拡散の開始状態を、

```math
u_{T'}=\frac{1}{2}y+\lambda_{add}\epsilon_{add},
\qquad \epsilon_{add}\sim\mathcal{CN}(0,1)
```

として構成する。これにより複素シンボル当たりの総noise分散は`3/4`となる。

## 新しく解釈・仮定した点

1. `P=1`は既存`ADJSCC/util_channel.py`に合わせ、実数要素当たりではなく**複素シンボル当たりの平均電力1**と解釈した。したがって、`CN(0,sigma^2)`の実部と虚部の分散はそれぞれ`sigma^2/2`である。
2. 実数latentは既存Channelと同じくNHWC全体をflattenし、前半を複素信号の実部、後半を虚部として扱う。
3. `gamma_ch`は物理AWGNのSNRであり、Encoderへ渡す`low_snr`およびDecoderへ渡す`high_snr`とは独立である。
4. 理論的な下限は約`-4.771 dB`だが、指示に従い、入力検査はより保守的な`channel_snr_db >= -4.0 dB`とした。`-4.0 dB`未満では式を評価せず`ValueError`を返す。
5. Resfusion checkpointの`T`とscheduleを使用する。ただしnoise completion式は`alpha_bar=1/4`を仮定するため、checkpointの開始点が`sqrt(alpha_bar)≈1/2`であることを前提とする。
6. 既存Resfusionは、電力正規化済みraw latentを学習データ共通のmin/maxで`[-1,1]`へ写像して学習している。本テストでは物理領域でAWGNとnoise completionを適用した後、同じaffine写像で`y`と`u_T'`をnetwork座標へ変換する。この写像ではclipしない。
7. 上記affine写像にはoffsetがあり、raw潜在空間の分散式とnetwork内部の拡散分散が厳密に同一とは限らない。さらに現在のResfusionはclean low latentを条件として学習されており、noisy `y`を条件とする学習は行われていない。したがって、本ファイルは提案初期化の**推論実験**であり、提案システムに厳密対応した学習済みモデルではない。
8. 逆拡散後の推定high latentは、Decoderへ渡す直前に`P=1`へ再電力正規化する。
9. 物理AWGN、追加noise、逆拡散各stepのnoiseは、seedから生成する。追加noiseと逆拡散noiseは物理AWGNから独立である。公平な比較のため、提案法とactual-channel ADJSCC baselineには同じ標準複素AWGN標本を使う。ただしEncoder出力が異なるため、両者の受信信号そのものは異なる。
10. 評価画像は256角center cropとし、全画像終了後にPSNR、LPIPS、DISTS、FIDを計測する。

## 実行方法

物理チャネルSNRが`0 dB`の場合：

```bash
python test_awgn_adjscc_signal_resfusion.py \
  --resfusion_ckpt my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt \
  --channel_snr_db 0 \
  --input_dir ../datasets/Raindrop/test_a/gt \
  --output_dir my_awgn_resfusion_eval_snr0
```

`channel_snr_db < -4.0`はエラーとなる。

```bash
python test_awgn_adjscc_signal_resfusion.py \
  --resfusion_ckpt path/to/model.ckpt \
  --channel_snr_db -5
```

出力：

```text
my_awgn_resfusion_eval_snr0/
  awgn_noise_completion/*_reconstructed.png
  no_resfusion_low_decoder/*_reconstructed.png
  no_resfusion_high_decoder/*_reconstructed.png
  actual_channel_adjscc/*_reconstructed.png
  metrics.csv
  summary_metrics.csv
```

`metrics.csv`には4経路の画像ごとのPSNRを保存する。`summary_metrics.csv`には各経路について、物理SNR、`P`、channel/additional noiseの複素標準偏差、平均PSNR、平均LPIPS、平均DISTS、FIDを保存する。なおFIDは画像ごとの平均ではなく、GT画像集合と各再構成画像集合の特徴分布間距離である。

## 重要な制約

現在のcheckpointをそのまま使う場合、Resfusionはnoisy条件信号`y`で学習されていない。この推論テストで有効性を確認できない場合、学習側も各sampleで物理AWGNを生成し、同じnoise completion初期化とnoisy conditionを用いるよう変更する必要がある。

python test_awgn_adjscc_signal_resfusion.py \
    --resfusion_ckpt my_resfusion_train/lightning_logs/version_0/checkpoints/last.ckpt \
    --channel_snr_db 0 \
    --input_dir ../datasets/Raindrop/test_a/gt \
    --output_dir my_awgn_resfusion_eval_snr0
