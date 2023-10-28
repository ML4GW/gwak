import os
import argparse
import numpy as np
import torch

hists = snakemake.input
save_path = snakemake.params[0]

new_hist = np.zeros((np.load(hists[0]).shape))
for hist in hists:
    past_hist = np.load(hist)
    new_hist += past_hist

np.save(save_path, new_hist)