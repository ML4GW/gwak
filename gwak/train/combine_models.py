#!/usr/bin/env python3
import torch
import torch.nn as nn
import argparse
import yaml

class CombinedModel(nn.Module):
    def __init__(self, embedder_model, metric_model):
        super().__init__()
        self.embedder_model = embedder_model
        self.metric_model = metric_model

    def forward(self, x):
        # Get the embedding from the embedder model.
        embedding = self.embedder_model(x)
        c = self.frequency_cos_similarity(x)
        # Pass the embedding to the metric model to get final classification.
        out = self.metric_model(embedding, c)
        return out

    def frequency_cos_similarity(self, batch):
        H = torch.fft.rfft(batch[:, 0, :], dim=-1)
        L = torch.fft.rfft(batch[:, 1, :], dim=-1)
        numerator = torch.sum(H * torch.conj(L), dim=-1)
        norm_H = torch.linalg.norm(H, dim=-1)
        norm_L = torch.linalg.norm(L, dim=-1)
        rho_complex = numerator / (norm_H * norm_L + 1e-8)
        rho_real = torch.real(rho_complex).unsqueeze(-1)
        return rho_real

def main(embedder_model_file,
         metric_model_file,
         batch_size=256,
         kernel_length=0.5,
         sample_rate=4096,
         num_ifos=2,
         output_path="model_JIT.pt"):

    # Load the embedder model (TorchScript traced module)
    embedder_model = torch.jit.load(embedder_model_file, map_location="cpu")
    embedder_model = embedder_model.to("cuda:0")
    embedder_model.eval()

    # Load the metric model (TorchScript traced module)
    metric_model = torch.jit.load(metric_model_file, map_location="cpu")
    metric_model = metric_model.to("cuda:0")
    metric_model.eval()

    # Create the combined model
    combined_model = CombinedModel(embedder_model, metric_model).to("cuda:0")
    combined_model.eval()

    # Prepare a dummy input for tracing
    dummy_input = torch.randn(batch_size, num_ifos, int(kernel_length * sample_rate), device='cuda:0')

    # Trace the model (instead of scripting)
    traced_model = torch.jit.trace(combined_model, dummy_input)

    # Save the traced model
    traced_model.save(output_path)
    print(f"Combined model saved to {output_path}")

    # Test inference
    output = combined_model(dummy_input)
    print("Test inference complete.")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge JIT embedder and metric models files, save to JIT model.'
    )
    parser.add_argument('folder_embedder', type=str, help='Path to the folder containing JIT embedder model.')
    parser.add_argument('folder_metric', type=str, help='Path to the folder containing JIT metric model.')
    parser.add_argument('--config', type=str)
    parser.add_argument('--outfile', type=str, default='model_JIT.pt', help='Output file name for the JIT combined model (default: model_JIT.pt).')

    args = parser.parse_args()

    # Load the YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract values
    sample_rate = config['data']['init_args']['sample_rate']
    kernel_length = config['data']['init_args']['kernel_length']
    batch_size = 64  # You can make this configurable if needed

    main(
        embedder_model_file=args.folder_embedder,
        metric_model_file=args.folder_metric,
        batch_size=batch_size,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        num_ifos=2,
        output_path=args.outfile
    )