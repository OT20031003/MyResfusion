"""received-y標準Resfusionの式・独立noise・backward smoke test。"""

import numpy as np
import torch
from torch import nn

from adjscc_signal_data_standard import (
    StandardReceivedSignalDataset,
    from_model_domain,
    to_model_domain,
)
from model.adjscc_signal_resfusion_standard import StandardReceivedADJSCCResfusion
from test_awgn_adjscc_signal_resfusion_standard import standard_resfusion_initialization
from variance_scheduler import LinearProScheduler


class RecordingDenoiser(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1)
        self.last_condition = None

    def forward(self, x, time, input_cond=None, channel_snr_db=None):
        self.last_condition = input_cond
        return self.conv(torch.cat((x, input_cond), dim=1))


def make_cache():
    torch.manual_seed(7)
    return {
        "low": torch.randn(2, 4, 4, 4).half(),
        "high": torch.randn(2, 4, 4, 4).half(),
        "images": torch.randint(0, 256, (2, 16, 16, 3), dtype=torch.uint8),
    }


def make_batch():
    dataset = StandardReceivedSignalDataset(make_cache(), -3.0, 3.0, -4.0, 8.0)
    samples = [dataset[0], dataset[1]]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def make_model():
    return StandardReceivedADJSCCResfusion(
        RecordingDenoiser(4), LinearProScheduler(12),
        mode="epsilon", loss_type="L2", latent_min=-3.0, latent_max=3.0,
        blr=1e-3, batch_size=2, accum_iter=1, devices=1, num_nodes=1,
        weight_decay=0.0, optimizer_type="Adam",
        lr_scheduler_type="CosineAnnealingLR", epochs=1, min_lr=0.0,
    )


def test_received_y_residual_forward_formula_and_independent_noise():
    batch, model = make_batch(), make_model()
    terms = model.make_training_terms(batch)
    assert torch.equal(terms["residual"], batch["y"] - batch["s_high"])
    assert not torch.equal(terms["residual"], batch["s_low"] - batch["s_high"])
    expected = (
        torch.sqrt(terms["alpha_bar_t"]) * batch["s_high"]
        + (1.0 - torch.sqrt(terms["alpha_bar_t"])) * (batch["y"] - batch["s_high"])
        + torch.sqrt(1.0 - terms["alpha_bar_t"]) * terms["diffusion_epsilon"]
    )
    assert torch.allclose(terms["x_t"], expected)
    assert torch.all((terms["t"] >= 0) & (terms["t"] < model.T_acc))
    assert torch.all(batch["channel_snr_db"] >= -4.0)
    assert terms["diffusion_epsilon"].data_ptr() != batch["channel_epsilon"].data_ptr()
    assert torch.isfinite(terms["x_t"]).all()
    assert torch.isfinite(terms["target_resnoise"]).all()


def test_condition_is_y_and_backward_is_finite():
    batch, model = make_batch(), make_model()
    loss, prediction, _terms = model._loss_and_prediction(batch)
    assert model.denoising_module.last_condition is batch["y"]
    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
    loss.backward()
    assert model.denoising_module.conv.weight.grad is not None


def test_model_domain_roundtrip_has_no_clamp():
    raw = torch.tensor([-10.0, 0.0, 10.0])
    model = to_model_domain(raw, -1.0, 1.0)
    assert torch.any(model.abs() > 1.0)
    assert torch.allclose(from_model_domain(model, -1.0, 1.0), raw)


def test_dataset_rejects_below_minus_four_and_keeps_out_of_range_y():
    try:
        StandardReceivedSignalDataset(make_cache(), -0.1, 0.1, -4.01, 8.0)
    except ValueError:
        pass
    else:
        raise AssertionError("channel SNR below -4 dB was accepted")
    fixed = StandardReceivedSignalDataset(
        make_cache(), -0.1, 0.1, -4.0, -4.0, deterministic_seed=99
    )[0]
    assert torch.any(fixed["y"].abs() > 1.0)
    assert not torch.equal(fixed["y"], fixed["s_low"])


def test_standard_initialization_exact_formula_without_completion_variance():
    received = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 10.0
    alpha_bar = 0.37
    rng_a = np.random.default_rng(123)
    initial, diffusion_epsilon = standard_resfusion_initialization(
        received, alpha_bar, rng_a
    )
    received_complex = received.reshape(1, -1)
    half = received_complex.shape[1] // 2
    received_complex = received_complex[:, :half] + 1j * received_complex[:, half:]
    expected_complex = (
        np.sqrt(alpha_bar) * received_complex
        + np.sqrt(1.0 - alpha_bar) * diffusion_epsilon
    )
    expected = np.concatenate([expected_complex.real, expected_complex.imag], axis=1).reshape(received.shape)
    assert np.allclose(initial, expected)
