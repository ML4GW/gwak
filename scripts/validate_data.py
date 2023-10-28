import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import VERSION


def main(args):

    fig, ax = plt.subplots(
        nrows=len(args.datasets) + 3,
        ncols=2,
        figsize=(15, len(args.datasets) * 5))

    with open(f'data/{VERSION}/info.txt', 'w') as f:
        i = 0
        for dataset_file in args.datasets:

            data = np.load(dataset_file)
            dataname = os.path.basename(dataset_file).strip('.npy')

            if 'data' in data.keys():
                dataset = data['data']

                print(f'{dataset_file}')
                print(f'Dataset {dataname} shape is {dataset.shape}')
                f.write(f'{dataset_file} \n')
                f.write(f'Dataset {dataname} shape is {dataset.shape} \n')

                ax[i][0].plot(dataset[0, 0].flatten())
                ax[i][1].plot(dataset[0, 1].flatten())
                ax[i][0].set_title(f'{dataset_file}')
            else:

                dataset = data['noisy']
                print(f'{dataset_file} noisy')
                f.write(f'{dataset_file} noisy \n')
                f.write(f'Dataset {dataname} shape is {dataset.shape} \n')

                ax[i][0].plot(dataset[0, 0, 0, :].flatten(), label='noisy')
                ax[i][1].plot(dataset[0, 0, 1, :].flatten(), label='noisy')
                ax[i + 1][0].plot(dataset[-1, 0, 0,
                                          :].flatten(), label='noisy')
                ax[i + 1][1].plot(dataset[-1, 0, 1,
                                          :].flatten(), label='noisy')
                ax[i][0].set_title(f'{dataset_file} first batch')
                ax[i + 1][0].set_title(f'{dataset_file} last batch')

                dataset = data['clean']
                print(f'{dataset_file} clean')
                f.write(f'{dataset_file} clean \n')
                f.write(f'Dataset {dataname} shape is {dataset.shape} \n')

                ax[i][0].plot(dataset[0, 0, 0, :].flatten(), label='clean')
                ax[i][1].plot(dataset[0, 0, 1, :].flatten(), label='clean')
                ax[i + 1][0].plot(dataset[-1, 0, 0,
                                          :].flatten(), label='clean')
                ax[i + 1][1].plot(dataset[-1, 0, 1,
                                          :].flatten(), label='clean')

                ax[i][0].legend()
                ax[i + 1][0].legend()

                i += 1

            i += 1

    fig.savefig(f'data/{VERSION}/info.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('datasets', nargs='+', type=str,
                        help='Path to the Omicron output')
    args = parser.parse_args()
    main(args)
