import os
import cv2
import numpy as np

from torch.utils.data import Dataset

class AudioDataset_gray(Dataset):
    def __init__(self, root_dirs):
        self.root_dirs = root_dirs
        fake_paths = [os.path.join(rd, "fake_audio") for rd in root_dirs]
        real_paths = [os.path.join(rd, "real_audio") for rd in root_dirs]

        fake_image_paths = [os.listdir(fp) for fp in fake_paths]
        real_image_paths = [os.listdir(rp) for rp in real_paths]

        all_fakes = [[os.path.join(fp, image) for image in fake_image_paths[e]] for e, fp in enumerate(fake_paths)]
        all_reals = [[os.path.join(fp, image) for image in real_image_paths[e]] for e, fp in enumerate(real_paths)]

        all_fake_paths = [image_path for fake_images_sublist in all_fakes for image_path in fake_images_sublist]
        all_real_paths = [image_path for real_images_sublist in all_reals for image_path in real_images_sublist]

        all_fake_paths = sorted(all_fake_paths, key=os.path.getmtime)
        all_real_paths = sorted(all_real_paths, key=os.path.getmtime)

        self.x = all_fake_paths + all_real_paths
        self.y = [1 for _ in all_fake_paths] + [0 for _ in all_real_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            image_path = self.x[idx]
            image_class = self.y[idx]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)[np.newaxis, ...] / 255

            return image, image_class
        except:
            return np.zeros((400,400))[np.newaxis, ...], -1
    
class AudioDataset_RGB(Dataset):
    def __init__(self, root_dirs):
        self.root_dirs = root_dirs
        fake_paths = [os.path.join(rd, "fake_audio") for rd in root_dirs]
        real_paths = [os.path.join(rd, "real_audio") for rd in root_dirs]

        fake_image_paths = [os.listdir(fp) for fp in fake_paths]
        real_image_paths = [os.listdir(rp) for rp in real_paths]

        all_fakes = [[os.path.join(fp, image) for image in fake_image_paths[e]] for e, fp in enumerate(fake_paths)]
        all_reals = [[os.path.join(fp, image) for image in real_image_paths[e]] for e, fp in enumerate(real_paths)]

        all_fake_paths = [image_path for fake_images_sublist in all_fakes for image_path in fake_images_sublist]
        all_real_paths = [image_path for real_images_sublist in all_reals for image_path in real_images_sublist]

        all_fake_paths = sorted(all_fake_paths, key=os.path.getmtime)
        all_real_paths = sorted(all_real_paths, key=os.path.getmtime)

        self.x = all_fake_paths + all_real_paths
        self.y = [1 for _ in all_fake_paths] + [0 for _ in all_real_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            image_path = self.x[idx]
            image_class = self.y[idx]

            image = cv2.imread(image_path).astype(np.float32) / 255
            image = np.moveaxis(image, 2, 0)
            
            return image, image_class
        except:
            return np.zeros((3, 400,400)), -1

class AudioDataset_multi(Dataset):
    def __init__(self, root_dirs):
        self.root_dirs = root_dirs
        directories = [[os.path.join(rd,d) for d in os.listdir(rd)] for rd in root_dirs]

        fake_dicts = []
        real_dicts = []
        for folders in directories:
            fake_dict = {}
            real_dict = {}
            for features in sorted(folders):
                fake_folder_path = os.path.join(features, "fake_audio")
                fake_image_paths = [os.path.join(fake_folder_path, image_path) for image_path in os.listdir(fake_folder_path)]
                real_folder_path = os.path.join(features, "real_audio")
                real_image_paths = [os.path.join(real_folder_path, image_path) for image_path in os.listdir(real_folder_path)]

                fake_image_paths = sorted(fake_image_paths, key=os.path.getmtime)
                real_image_paths = sorted(real_image_paths, key=os.path.getmtime)

                fake_dict[os.path.split(features)[1]] = fake_image_paths
                real_dict[os.path.split(features)[1]] = real_image_paths

            fake_dicts.append(fake_dict)
            real_dicts.append(real_dict)

        all_fakes = {}
        for d in fake_dicts:
            for k in d.keys():
                if k in all_fakes:
                    all_fakes[k] = all_fakes[k] + d[k]
                else:
                    all_fakes[k] = d[k]

        all_reals = {}
        for d in real_dicts:
            for k in d.keys():
                if k in all_reals:
                    all_reals[k] = all_reals[k] + d[k]
                else:
                    all_reals[k] = d[k]

        real_multi_feature_paths = list(zip(*all_reals.values()))
        fake_multi_feature_paths = list(zip(*all_fakes.values()))

        self.x = fake_multi_feature_paths + real_multi_feature_paths
        self.y = [1 for _ in fake_multi_feature_paths] + [0 for _ in real_multi_feature_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            feature_paths = self.x[idx]
            image_class = self.y[idx]

            image = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for path in feature_paths])

            return image, image_class
        except:
            return np.zeros((5, 400,400)), -1

def import_dataset(dataset_type, root_dir):
    if dataset_type == "gray":
        return AudioDataset_gray(root_dir)
    elif dataset_type == "RGB":
        return AudioDataset_RGB(root_dir)
    elif dataset_type == "multi":
        return AudioDataset_multi(root_dir)