import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import LinearModel

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    GPU_NAME,
    SVM_LR,
    N_SVM_EPOCHS,
    FACTORS_NOT_USED_FOR_FM
    )
DEVICE = torch.device(GPU_NAME)


def optimize_hyperplane(signals, backgrounds):
    # saved as a batch, which needs to be flattened out
    backgrounds = np.reshape(backgrounds, (backgrounds.shape[
                             0] * backgrounds.shape[1], backgrounds.shape[2]))

    sigs = torch.from_numpy(signals).float().to(DEVICE)
    bkgs = torch.from_numpy(backgrounds).float().to(DEVICE)
    network = LinearModel(n_dims=sigs.shape[2]).to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=SVM_LR)

    for epoch in range(N_SVM_EPOCHS):
        optimizer.zero_grad()
        background_MV = network(bkgs)
        signal_MV = network(sigs)

        # second index are the indicies
        signal_MV = torch.min(signal_MV, dim=1)[0]
        zero = torch.tensor(0).to(DEVICE)

        background_loss = torch.maximum(
            zero,
            1 - background_MV).mean()
        signal_loss = torch.maximum(
            zero,
            1 + signal_MV).mean()

        loss = background_loss + signal_loss
        if epoch % 50 == 0:
            print(network.layer.weight.data.cpu().numpy()[0])
            print(loss.item())
        loss.backward()
        optimizer.step()

    torch.save(network.state_dict(), args.fm_model_path)
    return np.zeros(5)
    return network.layer.weight.data.cpu().numpy()[0]


def main(args):
    '''
    Fit a hyperplane to distinguish background, signal classes

    signals: shape (N_samples, time_axis, 5)
    backgrounds: shape (time_axis, 5)
    '''
    signal_evals = []
    midp_map = {
        'bbh_fm_optimization_evals.npy': 1440,
        'sglf_fm_optimization_evals.npy': 1440,
        'sghf_fm_optimization_evals.npy': 1440,
        'wnbhf_fm_optimization_evals.npy': 1850,
        'wnblf_fm_optimization_evals.npy': 1850,
        'supernova_fm_optimization_evals.npy': 2150}

    for file_name in args.signal_path:
        mid = midp_map[file_name.split("/")[-1]]
        signal_evals.append(np.load(f'{file_name}')[:, mid - 150:mid + 150, :])

    signal_evals = np.concatenate(signal_evals, axis=0)
    signal_evals = np.delete(signal_evals, FACTORS_NOT_USED_FOR_FM, -1)

    timeslide_evals = []
    for file_name in os.listdir(args.timeslide_path):
        if '.npy' in file_name:
            timeslide_evals.append(
                np.load(os.path.join(args.timeslide_path, file_name)))

    norm_factors = []
    for file_name in os.listdir(args.norm_factor_path):
        if '.npy' in file_name:
            norm_factors.append(
                np.load(os.path.join(args.norm_factor_path, file_name)))

    norm_factors = np.array(norm_factors)
    means = np.mean(norm_factors[:, 0, 0, :], axis=0)
    means = np.delete(means, FACTORS_NOT_USED_FOR_FM, -1)

    stds = np.mean(norm_factors[:, 1, 0, :], axis=0)
    stds = np.delete(stds, FACTORS_NOT_USED_FOR_FM, -1)

    timeslide_evals = np.concatenate(timeslide_evals, axis=0)
    timeslide_evals = np.delete(timeslide_evals, FACTORS_NOT_USED_FOR_FM, -1)

    signal_evals = (signal_evals - means) / stds
    timeslide_evals = (timeslide_evals - means) / stds

    np.save(args.norm_factor_save_file, np.stack([means, stds], axis=0))

    optimal_coeffs = optimize_hyperplane(signal_evals, timeslide_evals)
    np.save(args.save_file, optimal_coeffs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('save_file', type=str,
                        help='Where to save the best final metric parameters')
    parser.add_argument('fm_model_path', type=str,
                        help='Where to save the best final metric model')
    parser.add_argument('norm_factor_save_file', type=str,
                        help='Where to save the normalization factors')
    parser.add_argument('--timeslide-path', type=str,
                        help='list[str] pointing to timeslide files ')
    parser.add_argument('--signal-path', type=str, nargs='+',
                        help='list[str] pointing to signal files')
    parser.add_argument('--norm-factor-path', type=str,
                        help='list[str] pointing to norm factors')

    args = parser.parse_args()
    main(args)
