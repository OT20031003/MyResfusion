"""WR.tex の各 boxed equation に対する回帰 test。"""

import torch
from torch import nn
from adjscc_signal_data_wr import WRSignalDataset
from model.adjscc_signal_resfusion_wr import WRADJSCCSignalResfusion
from variance_scheduler import LinearProScheduler

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(0.0)); self.condition = None
    def forward(self, x, time, input_cond=None, channel_snr_db=None):
        self.condition = input_cond
        return torch.zeros_like(x) + self.weight

def model():
    return WRADJSCCSignalResfusion(Denoiser(), LinearProScheduler(12), mode="epsilon",
        loss_type="L2", blr=1e-3, batch_size=2, accum_iter=1, devices=1, num_nodes=1,
        weight_decay=0., optimizer_type="Adam", lr_scheduler_type="CosineAnnealingLR",
        epochs=1, min_lr=0.)

def batch():
    cache = {"low": torch.randn(2, 4, 3, 3).half(), "high": torch.randn(2, 4, 3, 3).half(),
             "images": torch.zeros(2, 12, 12, 3, dtype=torch.uint8)}
    dataset = WRSignalDataset(cache, -4.0, 10.0, deterministic_seed=7)
    samples = [dataset[0], dataset[1]]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}

def test_wr_training_equations_and_condition():
    value, net = batch(), model()
    terms = net.make_training_terms(value)
    a, root = terms["alpha_bar"], torch.sqrt(terms["alpha_bar"])
    sigma = value["sigma"].reshape(-1, 1, 1, 1)
    expected_z = ((2 * root - 1) * value["z_high"] + (1 - root) * value["y"]
                  + torch.sqrt(terms["completion_variance"].clamp_min(0)) * terms["epsilon_add"])
    expected_epsilon = ((1-root)*sigma*value["channel_noise"]
                        + torch.sqrt(terms["completion_variance"].clamp_min(0))*terms["epsilon_add"]) / torch.sqrt(1-a)
    assert torch.allclose(terms["z_t"], expected_z)
    assert torch.allclose(terms["epsilon"], expected_epsilon)
    assert torch.equal(terms["residual"], value["z_low"] - value["z_high"])
    loss, _, _ = net._loss_and_prediction(value)
    assert net.denoising_module.condition is value["y"]
    loss.backward(); assert net.denoising_module.weight.grad is not None

def test_alpha_bar_zero_and_start_point():
    net = model()
    assert net.alphas_hat_t_minus_1[0] == 1
    assert net.betas_hat[0] == 0
    assert torch.isclose(net.alphas_hat[net.T_acc-1], torch.tensor(.25), atol=1e-4)
