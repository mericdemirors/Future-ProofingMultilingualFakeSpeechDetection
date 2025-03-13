import os
import cv2
import numpy as np
from tqdm import tqdm

datasets = "/hdd/AGAC/AGAC2_datasets/images"
datasets = sorted([os.path.join(datasets, dataset) for dataset in os.listdir(datasets)])
corrupted = []

for e, path_to_check in enumerate(datasets):
    feature_folders = sorted([os.path.join(path_to_check, x) for x in os.listdir(path_to_check)])

    print(f"read all images and check their shapes and if there is any Nans or infs {e+1} out of {len(datasets)}")
    for ff in feature_folders:
        for cf in ["fake_audio", "real_audio"]:
            image_folder_path = os.path.join(ff,cf)
            print("checking", image_folder_path)
            image_paths = [os.path.join(image_folder_path, i) for i in os.listdir(image_folder_path)]
            for ip in tqdm(image_paths):
                try:
                    img = cv2.imread(ip)
                    if ((img.shape != (400,400,3)) or (np.isnan(img).sum() != 0) or (np.isinf(img).sum() != 0)):
                        corrupted.append(ip)
                except:
                    corrupted.append(ip)

print("total of", len(corrupted), "corrupted files")
with open(r'corrupteds.txt', 'w') as fp:
    for corrupted_file in corrupted:
        fp.write(corrupted_file + "\n")
