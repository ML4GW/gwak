import os
import numpy as np
import argparse
import torch

from helper_functions import mae_torch, freq_loss_torch
from models import LSTM_AE, LSTM_AE_SPLIT, DUMMY_CNN_AE, FAT

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (NUM_IFOS,
                    SEG_NUM_TIMESTEPS,
                    BOTTLENECK,
                    MODEL,
                    FACTOR,
                    GPU_NAME,
                    RECREATION_LIMIT)


def quak_eval(data, model_path, device, reduce_loss=True):
    # data required to be torch tensor at this point

    # check if the evaluation has to be done for one model or for several
    loss = dict()
    if not reduce_loss:
        loss['original'] = dict()
        loss['recreated'] = dict()
        loss['loss'] = dict()
        loss['freq_loss'] = dict()

    for dpath in model_path:
        coherent_loss = False
        if dpath.split("/")[-1] in ['bbh.pt', 'sglf.pt', 'sghf.pt']:
            coherent_loss = True

        model_name = dpath.split("/")[-1].split(".")[0]
        if MODEL[model_name] == "lstm":
            model = LSTM_AE_SPLIT(num_ifos=NUM_IFOS,
                                  num_timesteps=SEG_NUM_TIMESTEPS,
                                  BOTTLENECK=BOTTLENECK[model_name]).to(device)
        elif MODEL[model_name] == "dense":
            model = FAT(num_ifos=NUM_IFOS,
                        num_timesteps=SEG_NUM_TIMESTEPS,
                        BOTTLENECK=BOTTLENECK[model_name]).to(device)

        model.load_state_dict(torch.load(dpath, map_location=GPU_NAME))
        if reduce_loss:
            if coherent_loss:
                loss[os.path.basename(dpath)[:-3]] = \
                    freq_loss_torch(data, model(data).detach())
            elif not coherent_loss:
                loss[os.path.basename(dpath)[:-3]] = \
                    freq_loss_torch(data, model(data).detach())
        elif not reduce_loss:
            if coherent_loss:
                loss['loss'][os.path.basename(dpath)[:-3]] = \
                    mae_torch(data, model(
                        data).detach()).cpu().numpy()
                loss['freq_loss'][os.path.basename(dpath)[:-3]] = \
                    freq_loss_torch(data, model(data).detach())

            elif not coherent_loss:
                loss['loss'][os.path.basename(dpath)[:-3]] = \
                    mae_torch(data, model(
                        data).detach()).cpu().numpy()
                loss['freq_loss'][os.path.basename(dpath)[:-3]] = \
                    freq_loss_torch(data, model(data).detach())
            loss['original'][os.path.basename(
                dpath)[:-3]] = data[:RECREATION_LIMIT].cpu().numpy()
            loss['recreated'][os.path.basename(
                dpath)[:-3]] = model(data[:RECREATION_LIMIT]).detach().cpu().numpy()
    return loss


def main(args):

    DEVICE = torch.device(GPU_NAME)

    # load the data
    data = np.load(args.test_data)['data']
    data = torch.from_numpy(data).float().to(DEVICE)
    loss = quak_eval(data, args.model_path, DEVICE,
                     reduce_loss=args.reduce_loss)

    if args.reduce_loss:
        # move to CPU
        for key in loss.keys():
            loss[key] = loss[key].cpu().numpy()

    if args.save_file:
        np.savez(args.save_file, **loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        'test_data', help='Required path to the test data file')
    parser.add_argument('save_file', help='Required path to save the file to',
                        type=str)
    parser.add_argument('reduce_loss', help='Whether to reduce to loss values or return recreation',
                        type=str, default="False")

    parser.add_argument('--model-path', help='Required path to trained model',
                        nargs='+', type=str)
    args = parser.parse_args()
    args.reduce_loss = args.reduce_loss == "True"

    main(args)
