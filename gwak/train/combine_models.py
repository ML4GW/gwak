#!/usr/bin/env python3
import torch
import torch.nn as nn
import argparse

class CombinedModel(nn.Module):
    def __init__(self, embedder_model, metric_model):
        super().__init__()
        self.embedder_model = embedder_model
        self.metric_model = metric_model

    def forward(self, x):
        # Get the embedding from the embedder model.
        embedding = self.embedder_model(x)
        # Pass the embedding to the metric model to get final classification.
        out = self.metric_model(embedding)
        return out

def main(embedder_model_file,
         metric_model_file,
         batch_size=256,
         kernel_length=0.5,
         sample_rate=4096,
         num_ifos=2,
         output_path="model_JIT.pt"):

    # Load the embedder model (assumed to be a TorchScript module).
    embedder_model = torch.jit.load(embedder_model_file, map_location="cpu")
    embedder_model = embedder_model.to("cuda:0")
    embedder_model.eval()

    # Load the metric model (also in TorchScript format).
    metric_model = torch.jit.load(metric_model_file, map_location="cpu")
    metric_model = metric_model.to("cuda:0")
    metric_model.eval()
    # Create the combined model.
    combined_model = CombinedModel(embedder_model, metric_model)
    combined_model.eval()

    # Script the combined model to produce a single TorchScript module.
    scripted_model = torch.jit.script(combined_model)

    scripted_model.save(output_path)
    print(f"Combined model saved to {output_path}")

    dummy_input = torch.randn(batch_size, num_ifos, int(kernel_length * sample_rate), device='cuda:0')
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
    parser.add_argument('--outfile', type=str, default='Output folder for JIT combined model', help='Output file name for the NumPy array (default: far_metrics.npy).')

    args = parser.parse_args()

    # Load the YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract values
    sample_rate = config['data']['init_args']['sample_rate']
    kernel_length = config['data']['init_args']['kernel_length']
    psd_length = config['data']['init_args']['psd_length']
    fduration = config['data']['init_args']['fduration']
    fftlength = config['data']['init_args']['fftlength']
    batch_size = 64 #config['data']['init_args']['batch_size']
    batches_per_epoch = config['data']['init_args']['batches_per_epoch']
    num_workers = config['data']['init_args']['num_workers']
    data_saving_file = config['data']['init_args']['data_saving_file']
    signal_classes = config['data']['init_args']['signal_classes']

    main(
        embedder_model_file=args.folder_embedder,
        metric_model_file=args.folder_metric,
        batch_size=batch_size,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        num_ifos=2,
        output_path=args.outfile
    )
