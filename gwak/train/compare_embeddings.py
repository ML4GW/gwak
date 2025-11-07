import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from ml4gw.waveforms import SineGaussian, MultiSineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst
from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, MultiSineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from gwak.train.plotting import make_corner


class SimpleCNN1D(nn.Module):
    def __init__(self, input_channels=1, num_features=8, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * num_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_and_evaluate_classifier(X, y, num_features, embedder_name, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    model = SimpleCNN1D(input_channels=1, num_features=num_features, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), 32):
            indices = permutation[i:i+32]
            batch_x = X_train[indices]
            batch_y = y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        acc = (preds == y_test).float().mean().item()

    # ROC
    fpr, tpr, _ = roc_curve(y_test.numpy(), probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{embedder_name} (AUC={roc_auc:.2f})")

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate classifiers on embedder outputs.')
    parser.add_argument('embedders', nargs='+', help='List of embedder model paths')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--nevents', type=int)
    parser.add_argument('--output', type=str, required=True, help='Directory to save output plots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    sample_rate = config['data']['init_args']['sample_rate']
    kernel_length = config['data']['init_args']['kernel_length']
    psd_length = config['data']['init_args']['psd_length']
    fduration = config['data']['init_args']['fduration']
    fftlength = config['data']['init_args']['fftlength']
    batch_size = args.nevents
    num_workers = config['data']['init_args']['num_workers']

    signal_classes = [
        "MultiSineGaussian", "SineGaussian", "BBH", "Gaussian", "Cusp",
        "Kink", "KinkKink", "WhiteNoiseBurst", "CCSN", "Background", "Glitch", "FakeGlitch"
    ]

    priors = [
        MultiSineGaussianBBC(), SineGaussianBBC(), LAL_BBHPrior(), GaussianBBC(),
        CuspBBC(), KinkBBC(), KinkkinkBBC(), WhiteNoiseBurstBBC(),
        None, None, None, None
    ]
    waveforms = [
        MultiSineGaussian(sample_rate=sample_rate, duration=fduration + kernel_length),
        SineGaussian(sample_rate=sample_rate, duration=fduration + kernel_length),
        IMRPhenomPv2(),
        Gaussian(sample_rate=sample_rate, duration=fduration + kernel_length),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        GenerateString(sample_rate=sample_rate),
        WhiteNoiseBurst(sample_rate=sample_rate, duration=fduration + kernel_length),
        None, None, None, None
    ]
    extra_kwargs = [
        None, None, {"ringdown_duration": 0.9}, None, None, None, None, None,
        None, None, None, None
    ]

    loader = SignalDataloader(
        signal_classes, priors, waveforms, extra_kwargs,
        data_dir=args.data_dir,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        psd_length=psd_length,
        fduration=fduration,
        fftlength=fftlength,
        batch_size=batch_size,
        batches_per_epoch=2000000,
        num_workers=num_workers,
        ifos='HL',
        snr_prior=torch.distributions.Uniform(low=3, high=30),
        # glitch_root=f'/home/hongyin.chen/anti_gravity/gwak/gwak/output/omicron/HL/'
        glitch_root="/fred/oz016/Andy/New_Data/gwak/omicron/HL"
    )

    test_loader = loader.test_dataloader()
    clean_batch, glitch_batch = next(iter(test_loader))
    clean_batch = clean_batch.to(device)
    glitch_batch = glitch_batch.to(device)

    processed, labels, _ = loader.on_after_batch_transfer([clean_batch, glitch_batch], None, local_test=True)
    labels = labels.detach().cpu().numpy()
    binary_labels = (~np.isin(labels, [10, 11])).astype(int)

    results = []

    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (log scale)")

    for embed_path in args.embedders:
        print(f"\nEvaluating embedder: {embed_path}")
        embedder = torch.jit.load(embed_path).to(device)
        embedder.eval()

        with torch.no_grad():
            embeddings = embedder(processed).cpu().numpy()

        embedder_name = os.path.splitext(os.path.basename(embed_path))[0]
        fig = make_corner(
            embeddings,
            (labels - 1).astype(int),
            return_fig=True,
            label_names=signal_classes
        )
        fig.savefig(os.path.join(args.output, f"corner_{embedder_name}.png"))
        plt.close(fig)
        acc = train_and_evaluate_classifier(
            embeddings, binary_labels,
            num_features=embeddings.shape[1],
            embedder_name=embedder_name,
            output_dir=args.output
        )
        print(f"Accuracy: {acc*100:.2f}%")
        results.append((embed_path, acc))

    plt.plot([1e-6, 1], [1e-6, 1], 'k--')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(args.output, "roc_comparison_log.png")
    plt.savefig(roc_path)

    best_model = max(results, key=lambda x: x[1])
    print(f"\nBest embedder: {best_model[0]} with accuracy: {best_model[1]*100:.2f}%")