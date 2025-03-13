print("imports...", flush=True)
import os
import argparse

import pandas as pd
from tqdm import tqdm

from utils import *
import GLOBAL_VARIABLES

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-dataset_path', '-dp', type=str, default="", help='path to datasets "clean_data" folder. default is: ""')
    parser.add_argument('-path', '-p', type=str, default="", help='directory to save images. default is: ""')
    parser.add_argument('-start', '-s', type=int, default=0, help='index of first processed audio. default is: 0')
    parser.add_argument('-end', '-e', type=int, default=0, help='index of last processed audio. default is: 0')
    parser.add_argument('-noise_sigma', '-n', type=float, default=0, help='sigma value for additive gaussian noise. default is: 0')
    parser.add_argument('-segment_size', '-ss', type=int, default=400, help='segment(window) size to slide on audio. default is: 400')
    parser.add_argument('-segment_overlap', '-so', type=int, default=200, help='how much does each segment(window) overlap with eachother. default is: 200')
    return parser.parse_args()

args = parse_arguments()
save_folder_path = args.path
start = args.start
end = args.end

GLOBAL_VARIABLES.NOISE_SIGMA = args.noise_sigma
GLOBAL_VARIABLES.SEGMENT_SIZE = args.segment_size
GLOBAL_VARIABLES.SEGMENT_OVERLAP = args.segment_overlap
GLOBAL_VARIABLES.maxlag = int(GLOBAL_VARIABLES.SEGMENT_SIZE / 2)

real_audio_df = pd.read_csv(os.path.join(args.dataset_path, "real.csv"))
fake_audio_df = pd.read_csv(os.path.join(args.dataset_path, "fake.csv"))
if "WaveFake" in args.dataset_path:
    fake_audio_df = fake_audio_df[fake_audio_df["path"].str.contains("LJ")]
real_audio_df = real_audio_df.sort_values(["seconds"]).reset_index(drop=True)
fake_audio_df = fake_audio_df.sort_values(["seconds"]).reset_index(drop=True)

features = ["absolutes","angles","reals","imags", "cum3s"]
for f in features:
    os.makedirs(os.path.join(save_folder_path, f, "real_audio"), exist_ok=True)
    os.makedirs(os.path.join(save_folder_path, f, "fake_audio"), exist_ok=True)

print("real images...", flush=True)
for i in tqdm(range(start, min(len(real_audio_df), end))):
    path = real_audio_df["path"][i]
    image_name = path.split(os.sep)[-1][:-4]

    [absolute_path,angle_path,real_path,imag_path,cum3_path] = [os.path.join(save_folder_path, f, "real_audio" ,image_name + ".png") for f in features]

    if os.path.isfile(absolute_path) and os.path.isfile(angle_path) and os.path.isfile(real_path) and os.path.isfile(imag_path) and os.path.isfile(cum3_path):
        continue

    RC_layers, cum3_avg = get_features(path, max_K=-1)
    signature_image = create_signature_image(RC_layers)
    save_images(signature_image, cum3_avg, absolute_path, angle_path, real_path, imag_path, cum3_path)

for i in tqdm(range(start, min(len(fake_audio_df), end))):
    path = fake_audio_df["path"][i]
    image_name = path.split(os.sep)[-1][:-4]
    [absolute_path,angle_path,real_path,imag_path,cum3_path] = [os.path.join(save_folder_path, f, "fake_audio" ,image_name + ".png") for f in features]

    if os.path.isfile(absolute_path) and os.path.isfile(angle_path) and os.path.isfile(real_path) and os.path.isfile(imag_path) and os.path.isfile(cum3_path):
        continue

    RC_layers, cum3_avg = get_features(path, max_K=-1)
    signature_image = create_signature_image(RC_layers)
    save_images(signature_image, cum3_avg, absolute_path, angle_path, real_path, imag_path, cum3_path)