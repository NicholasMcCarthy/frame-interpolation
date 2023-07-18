import os
from pathlib import Path
import numpy as np
import tempfile
import tensorflow as tf
import mediapy
from PIL import Image

import natsort
import functools
from eval import interpolator, util

from tqdm import tqdm
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
import argparse

parser = argparse.ArgumentParser(description='FILM frame-interpolation')
parser.add_argument('--input_dir', required=True, type=str, help='Path to the input directory')
parser.add_argument('--output_dir', required=True, type=str, help='Path to the output directory')
parser.add_argument('--times_to_interpolate', default=5, type=int, help='Number of times to interpolate (default: 5)')

args = parser.parse_args()


def main():

    from eval import interpolator
    # import tensorflow as tf

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_path = "pretrained_models/film_net/Style/saved_model"
    output_video = True
    times_to_interpolate = 5

    INPUT_EXT = ['png', 'jpg', 'jpeg']

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    interpolator = interpolator.Interpolator("pretrained_models/film_net/Style/saved_model", None)

    # Batched time.
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

    # Get files from input dir
    input_frames = natsort.natsorted([x for x in os.listdir(input_dir) if
                           os.path.isfile(os.path.join(input_dir, x)) and os.path.splitext(x)[-1] == '.png'])

    input_frames = [os.path.join(input_dir, x) for x in input_frames]
    print(input_frames[:3])
    # input_frames_list = [natsort.natsorted(tf.io.gfile.glob(f'{input_dir}/*.{ext}')) for ext in INPUT_EXT]
    #
    # input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
    num_frames = len(input_frames)
    print(f'Generating {num_frames} in-between frames for {input_dir}')

    print(f'Making directory {output_dir}')
    tf.io.gfile.makedirs(output_dir)

    for i, frame in tqdm(enumerate(util.interpolate_recursively_from_files(
            input_frames, times_to_interpolate, interpolator))):
        util.write_image(f'{output_dir}/frame_{i:05d}.png', frame)


if __name__ == '__main__':
    main()
