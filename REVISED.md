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
- 既存 Resfusion と同じ posterior sampling を実装した。再現的な平均経路が必要な場合は `--deterministic` で各逆 step の posterior noise を 0 にできる。開始点の `epsilon_add` は常に生成する。

## コマンド例

学習:

```bash
python train_adjscc_signal_resfusion_wr.py \
  --adjscc_weights /path/to/adjscc.h5 \
  --data_dir ../datasets/Raindrop \
  --channel-snr-min-db -4.0 \
  --channel-snr-max-db 20.0 \
  --noise_schedule LinearPro --T 12
```

推論:

```bash
python infer_adjscc_signal_resfusion_wr.py \
  --resfusion_ckpt /path/to/wr-best.ckpt \
  --channel_snr_db 0 \
  --input_dir ../datasets/Raindrop/test_a/gt \
  --output_dir wr_signal_inference
```

テスト:

```bash
pytest -q test_adjscc_signal_resfusion_wr.py
```
