import torch
import torch.nn as nn
import os
from gwak.train.fm_models import BackgroundFlowModel  # adjust this import

class FlowWrapper(nn.Module):
    def __init__(self, flow, standardizer=None):
        super().__init__()
        self.flow = flow
        self.standardizer = standardizer

    def forward(self, x, context=None):
        if self.standardizer is not None:
            x = self.standardizer(x)
        return self.flow.log_prob(inputs=x, context=context) if context is not None else self.flow.log_prob(inputs=x)

# --- Manually initialize the model ---
model = BackgroundFlowModel(
    embedding_model="output/s4_kl1.0_bs256_HL/model_JIT.pt",
    ckpt=None,
    cfg_path="train/configs/NF_onlyBkg.yaml",
    means='output/s4_kl1.0_bs256_HL/means.npy',
    stds='output/s4_kl1.0_bs256_HL/stds.npy',
    new_shape=128,
    n_dims=10,
    n_flow_steps=4,
    hidden_dim=64,
    learning_rate=1e-3,
    use_freq_correlation=False
)

# --- Load weights from checkpoint ---
ckpt_path = "output/s4_kl1.0_bs256_NF_onlyBkg_HL/lightning_logs/c6p9c70k/checkpoints/last.ckpt"
state_dict = torch.load(ckpt_path, map_location="cuda:0")["state_dict"]
model.load_state_dict(state_dict, strict=False)  # strict=False in case some keys mismatch
model.model.eval().to("cuda:0")

# --- Wrap and trace ---
wrapper = FlowWrapper(model.model, standardizer=model.standardizer).to("cuda:0").eval()
input_dim = model.model._transform._transforms[0].features
example_input = torch.randn(1, input_dim).to("cuda:0")
traced = torch.jit.trace(wrapper, example_input)
traced.save("output/s4_kl1.0_bs256_NF_onlyBkg_HL/model_JIT.pt")


