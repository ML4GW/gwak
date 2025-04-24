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
    # breakpoint()
    # dummy_input = torch.randn(1, 8).to("cuda")
    # metric_model = metric_model.to("cuda")
    # metric_model(dummy_input)
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
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference (default: 256).')
    parser.add_argument('--kernel_length', type=float, default=0.5, help='Kernel length in seconds (default: 0.5).')
    parser.add_argument('--sample_rate', type=float, default=4096, help='Sample rate in Hz (default: 4096).')
    parser.add_argument('--num_ifos', type=int, default=2, help='Number of interferometers (default: 2).')
    parser.add_argument('--outfile', type=str, default='Output folder for JIT combined model', help='Output file name for the NumPy array (default: far_metrics.npy).')
    
    args = parser.parse_args()

    main(
        embedder_model_file=args.folder_embedder,
        metric_model_file=args.folder_metric,
        batch_size=args.batch_size,
        kernel_length=args.kernel_length,
        sample_rate=args.sample_rate,
        num_ifos=args.num_ifos,
        output_path=args.outfile
    )

    # for KATYA this would get called as such: 
    # python combine_models.py /home/eric.moreno/gwak2/gwak/output/combination/embedding_model_JIT.pt 
    #                          /home/eric.moreno/gwak2/gwak/output/combination/mlp_model_JIT.pt 
    #       #from export.yaml: --batch_size 256 --kernel_length 0.5 --sample_rate 4096 --num_ifos 2
    #                          --outfile /home/eric.moreno/gwak2/gwak/output/combination/model_JIT.pt
