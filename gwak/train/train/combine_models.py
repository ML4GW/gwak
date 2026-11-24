#!/usr/bin/env python3
import torch
import torch.nn as nn
import argparse
import yaml
from transforms import cohmode_to_dim, frequency_cos_similarity
from transforms.preprocess import FeatureExtractor

class CombinedModel(nn.Module):
    def __init__(
        self,
        embedder_model,
        metric_model,
        feats_mode,
        feature_mode,
        apply_context,
        full_return=True
    ):
        super().__init__()
        self.embedder_model = embedder_model
        self.metric_model = metric_model
        self.full_return = full_return
        self.feature_mode = feature_mode
        self.apply_context = apply_context
        self.feature_extractor = FeatureExtractor(
            mode=feature_mode
        )

        if feats_mode == "embedding":
            self.inputs_module = embedder_model
            self.context_module = self.feature_extractor
        else:
            self.inputs_module = self.feature_extractor
            self.context_module = embedder_model

    def forward(self, x):
        # Get the embedding from the embedder model.
        inputs = self.inputs_module(x)
        # Pass the embeddin to the metric model to get final classification.
        if apply_context:
            context = self.context_module(x)
            out = self.metric_model(inputs, context)        
            inputs = torch.cat([inputs, context], dim=1)
        else:
            out = self.metric_model(inputs)

        if self.full_return:
            out = torch.cat([inputs, out.reshape(-1, 1)], dim=1)

        return out


def main(
    embedder_model_file,
    metric_model_file,
    embedding_size:int,
    feature_mode,
    feats_mode,
    apply_context,
    batch_size:int=256,
    kernel_length:float=0.5,
    sample_rate:int=4096,
    num_ifos:int=2,
    output_path="model_JIT.pt"
):

    # Load the embedder model (TorchScript traced module)
    embedder_model = torch.jit.load(embedder_model_file, map_location="cpu")
    embedder_model = embedder_model.to("cuda:0")
    embedder_model.eval()

    # Load the metric model (TorchScript traced module)
    metric_model = torch.jit.load(metric_model_file, map_location="cpu")
    metric_model = metric_model.to("cuda:0")
    metric_model.eval()

    # Create the combined model
    combined_model = CombinedModel(
        embedder_model,
        metric_model,
        feats_mode,
        feature_mode,
        apply_context,
    ).to("cuda:0")
    combined_model.eval()

    # Prepare a dummy input for tracing
    dummy_input = torch.randn(
        batch_size, num_ifos, 
        int(kernel_length * sample_rate), device='cuda:0'
    )

    # Trace the model (instead of scripting)
    traced_model = torch.jit.trace(combined_model, dummy_input)

    # Save the traced model
    traced_model.save(output_path)
    print(f"Combined model saved to {output_path}")

    # Test inference
    output = combined_model(dummy_input)
    print("Test inference complete.")
    print(output)
    print(output.sum())
    coh_size = cohmode_to_dim(feature_mode)

    # assert output.shape[-1] == (embedding_size + coh_size + 1), "Unentended output shape"
    # print(f"Output shape: {output.shape}")
    # print(f"Format: {[embedding_size, coh_size, 1]}")
    print(f"Output: {output[:,-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge JIT embedder and metric models files, save to JIT model.'
    )
    parser.add_argument('folder_embedder', type=str, help='Path to the folder containing JIT embedder model.')
    parser.add_argument('folder_metric', type=str, help='Path to the folder containing JIT metric model.')
    parser.add_argument('--feature_mode', type=str)
    parser.add_argument('--embd_config', type=str)
    parser.add_argument('--fm_config', type=str)
    parser.add_argument('--outfile', type=str, default='model_JIT.pt', help='Output file name for the JIT combined model (default: model_JIT.pt).')

    args = parser.parse_args()

    # Load the YAML config file
    with open(args.embd_config, 'r') as f:
        embd_config = yaml.safe_load(f)
    with open(args.fm_config, 'r') as f:
        fm_config = yaml.safe_load(f)

    # Extract values
    kernel_length = embd_config['data']['init_args']['kernel_length']
    sample_rate = embd_config['data']['init_args']['sample_rate']
    embedding_size = embd_config["model"]["init_args"]["d_output"]
    batch_size = 64  # You can make this configurable if needed
    feats_mode = fm_config["model"]["init_args"]["feats_mode"]
    apply_context = fm_config["model"]["init_args"]["apply_context"]

    main(
        embedder_model_file=args.folder_embedder,
        metric_model_file=args.folder_metric,
        embedding_size=embedding_size,
        feature_mode=args.feature_mode,
        feats_mode=feats_mode,
        apply_context=apply_context,
        batch_size=batch_size,
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        num_ifos=2,
        output_path=args.outfile
    )
