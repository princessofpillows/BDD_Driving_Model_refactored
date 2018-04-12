import h5py

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import get_config, print_usage
from utils.preprocessing import package_data


def main(config):
    """The main function."""

    # Package data from directory into HD5 format
    print("Packaging data...")
    package_data(config.data_dir)

    # Load packaged data
    print("Loading data...")
    f = h5py.File('videoData.h5', 'r')
    data = []
    for group in f:
        data.append(list(f[group]))
    

if __name__ == "__main__":

    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


