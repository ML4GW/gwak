import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from models import LinearModel
from evaluate_data import full_evaluation
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    FM_TIMESLIDE_TOTAL_DURATION,
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    RETURN_INDIV_LOSSES,
    FACTORS_NOT_USED_FOR_FM
)


def main(args):

    DEVICE = torch.device(f'cuda:{args.gpu}')

    if args.metric_coefs_path is not None:
        # initialize histogram
        n_bins = 2 * int(HISTOGRAM_BIN_MIN / HISTOGRAM_BIN_DIVISION)
        hist = np.zeros(n_bins)
        np.save(args.save_path, hist)

        # compute the dot product and save that instead
        metric_vals = np.load(args.metric_coefs_path)
        norm_factors = np.load(args.norm_factor_path)
        metric_vals = torch.from_numpy(metric_vals).float().to(DEVICE)
        norm_factors = torch.from_numpy(norm_factors).float().to(DEVICE)

        def update_hist(vals):
            vals = np.array(vals)
            # a trick to not to re-evaluate saved timeslides
            vals = np.delete(vals, FACTORS_NOT_USED_FOR_FM, -1)
            vals = torch.from_numpy(vals).to(DEVICE)
            # flatten batch dimension
            vals = torch.reshape(vals, (vals.shape[
                                         0] * vals.shape[1], vals.shape[2]))
            means, stds = norm_factors[0], norm_factors[1]
            vals = (vals - means) / stds

            if RETURN_INDIV_LOSSES:
                model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
                model.load_state_dict(torch.load(
                    args.fm_model_path, map_location=f'cuda:{args.gpu}'))
                vals = model(vals).detach()
            else:
                vals = torch.matmul(vals, metric_vals)

            update = torch.histc(vals, bins=n_bins,
                                 min=-HISTOGRAM_BIN_MIN, max=HISTOGRAM_BIN_MIN)
            past_hist = np.load(args.save_path)
            new_hist = past_hist + update.cpu().numpy()
            np.save(args.save_path, new_hist)

        # load pre-computed timeslides evaluations
        for folder in args.data_path:

            all_files = os.listdir(folder)
            print(f'Analyzing {folder} from {args.data_path}')

            for file_id in range(0,len(all_files)-len(all_files) % 5,5):

                if file_id%10000==0: print(f'Analyzing {file_id} from {len(all_files)}')
                all_vals = [ np.load(os.path.join(folder, all_files[file_id+local_id]))
                            for local_id in range(5)
                            if '.npy' in all_files[file_id+local_id] ]

                all_vals = np.concatenate(all_vals, axis=0)
                update_hist(all_vals)

    else:

        data = np.load(args.data_path[0])['data']
        data = torch.from_numpy(data).to(DEVICE)

        reduction = 20  # for things to fit into memory nicely

        timeslide_total_duration = TIMESLIDE_TOTAL_DURATION
        if args.fm_shortened_timeslides:
            timeslide_total_duration = FM_TIMESLIDE_TOTAL_DURATION

        sample_length = data.shape[1] / SAMPLE_RATE
        n_timeslides = int(timeslide_total_duration //
                           sample_length) * reduction
        print('Number of timeslides:', n_timeslides)

        for timeslide_num in range(1, n_timeslides + 1):
            print(f'starting timeslide: {timeslide_num}/{n_timeslides}')

            indicies_to_slide = np.random.uniform(
                SAMPLE_RATE, data.shape[1] - SAMPLE_RATE)
            indicies_to_slide = int(indicies_to_slide)
            timeslide = torch.empty(data.shape, device=DEVICE)

            # hanford unchanged
            timeslide[0, :] = data[0, :]

            # livingston slid
            timeslide[1, :indicies_to_slide] = data[1, -indicies_to_slide:]
            timeslide[1, indicies_to_slide:] = data[1, :-indicies_to_slide]

            # make a random cut with the reduced shape
            reduced_len = int(data.shape[1] / reduction)
            start_point = int(np.random.uniform(
                0, data.shape[1] - SAMPLE_RATE - reduced_len))
            timeslide = timeslide[:, start_point:start_point + reduced_len]

            timeslide = timeslide[:, :(timeslide.shape[1] // 1000) * 1000]
            final_values = full_evaluation(
                timeslide[None, :, :], args.model_folder_path, DEVICE)

            print('saving, individually')
            means, stds = torch.mean(
                final_values, axis=-2), torch.std(final_values, axis=-2)
            means, stds = means.detach().cpu().numpy(), stds.detach().cpu().numpy()
            np.save(f'{args.save_normalizations_path}/normalization_params_{timeslide_num}.npy', np.stack([means, stds], axis=0))
            final_values = final_values.detach().cpu().numpy()

            # save as a numpy file, with the index of timeslide_num
            np.save(f'{args.save_evals_path}/timeslide_evals_{timeslide_num}.npy', final_values)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('save_path', type=str,
                        help='Folder to which save the timeslides')

    parser.add_argument('model_folder_path', nargs='+', type=str,
                        help='Path to the folder containing the models')

    # Additional arguments
    parser.add_argument('--data-path', type=str, nargs='+',
                        help='Directory containing the timeslides')

    parser.add_argument('--fm-model-path', type=str,
                        help='Final metric model')

    parser.add_argument('--metric-coefs-path', type=str, default=None,
                        help='Pass in path to metric coefficients to compute dot product')

    parser.add_argument('--norm-factor-path', type=str, default=None,
                        help='Pass in path to significance normalization factors')

    parser.add_argument('--fm-shortened-timeslides', type=str, default='False',
                        help='Generate reduced timeslide samples to train final metric')

    parser.add_argument('--gpu', type=str, default='1',
                        help='On which GPU to run')

    parser.add_argument('--save-evals-path', type=str, default=None,
                        help='Where to save evals')

    parser.add_argument('--save-normalizations-path', type=str, default=None,
                        help='Where to save normalizations')

    args = parser.parse_args()
    args.fm_shortened_timeslides = args.fm_shortened_timeslides == 'True'

    main(args)
