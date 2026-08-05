"""AWGN付きADJSCC Resfusion学習経路の小規模回帰テスト。"""

import torch
from torch import nn

from adjscc_signal_data import AWGNSignalPairDataset
from model import ADJSCCSignalResfusion
from variance_scheduler import LinearProScheduler


class TinyDenoiser(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layer = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x, time, input_cond=None, channel_snr_db=None):
        assert channel_snr_db is not None and channel_snr_db.shape == (x.shape[0],)
        return self.layer(torch.cat((x, input_cond), dim=1))


def _cache():
    torch.manual_seed(10)
    return {
        "low": torch.randn(2, 4, 4, 4).half(),
        "high": torch.randn(2, 4, 4, 4).half(),
        "images": torch.randint(0, 256, (2, 16, 16, 3), dtype=torch.uint8),
        "names": ["a", "b"], "metadata": {},
    }


def test_awgn_dataset_has_no_post_noise_clamp_or_normalization():
    cache = _cache()
    dataset = AWGNSignalPairDataset(cache, -2.0, 2.0, -4.0, -4.0)
    sample = dataset[0]
    assert sample["s_low"].shape == sample["s_high"].shape == sample["y"].shape
    assert sample["channel_snr_db"].item() == -4.0
    # 低SNRなら固定model範囲を外れる受信値が残り、clampされていない。
    assert torch.any(sample["y"].abs() > 1.0)
    assert not torch.equal(sample["s_low"], sample["y"])


def test_training_uses_clean_residual_and_backward_succeeds():
    cache = _cache()
    dataset = AWGNSignalPairDataset(cache, -2.0, 2.0, 0.0, 5.0)
    samples = [dataset[0], dataset[1]]
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
    model = ADJSCCSignalResfusion(
        TinyDenoiser(4), LinearProScheduler(12), mode="epsilon", loss_type="L2",
        latent_min=-2.0, latent_max=2.0, blr=1e-3, batch_size=2,
        accum_iter=1, devices=1, num_nodes=1, weight_decay=0.0,
        optimizer_type="Adam", lr_scheduler_type="CosineAnnealingLR",
        epochs=1, min_lr=0.0,
    )
    terms = model.make_training_terms(batch)
    assert torch.all((0 <= terms["t"]) & (terms["t"] < model.T_acc))
    assert terms["x_t"].shape == batch["s_low"].shape
    assert torch.equal(terms["residual"], batch["s_low"] - batch["s_high"])
    assert not torch.equal(terms["residual"], batch["y"] - batch["s_high"])
    # eps_diffはdataset内のchannel AWGNとは別のtorch.randn_like呼出しで生成される。
    assert terms["eps_diff"].data_ptr() != batch["y"].data_ptr()
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.denoising_module.layer.weight.grad is not None


def test_rejects_channel_snr_below_minus_four_db():
    try:
        AWGNSignalPairDataset(_cache(), -2.0, 2.0, -4.01, 10.0)
    except ValueError:
        return
    raise AssertionError("-4 dB未満を拒否しませんでした")
