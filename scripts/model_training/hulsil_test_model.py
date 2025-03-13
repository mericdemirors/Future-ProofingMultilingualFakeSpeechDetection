print("imports...", flush=True)
import os
import argparse
import glob
from tqdm import tqdm

import torch
from datetime import datetime

# import warnings
# warnings.filterwarnings("ignore")

from datasets import *
from models import *

from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-model_type', '-m', type=str, default="mid_CNN", help='model type to train. default is: \"mid_CNN\"')
    parser.add_argument('-checkpoint_path', '-cp', type=str, default="", help='Checkpoint file to test. default is: \"\"')
    parser.add_argument('-root_dirs', '-r', type=str, default="", nargs='*', help='list of image dataset folders. default is: \"\"')
    parser.add_argument('-root_dirs_code', '-c', type=str, default="", help='code name for current train dataset. default is: \"\"')
    parser.add_argument('-dataset_type', '-d', type=str, default="gray", help='Dataset type to train on. default is: \"gray\"')
    parser.add_argument('-batch', '-b', type=int, default=32, help='Batch size. default is: 32')
    parser.add_argument('-threshold', '-th', type=float, default=0.5, help='Threshold for sigmoid classification. default is: 0.5')
    return parser.parse_args()

args = parse_arguments()
model_type = args.model_type
checkpoint_path = args.checkpoint_path
root_dirs = args.root_dirs
root_dirs_code = args.root_dirs_code
dataset_type = args.dataset_type
BATCH_SIZE = args.batch
THRESHOLD = args.threshold

model_serial_number = f"{model_type}_{os.path.split(root_dirs_code)[1]}_{datetime.now().strftime('%m:%d:%H:%M:%S')}"
print("serial_number:", model_serial_number, flush=True)
model_serial_path = os.path.join("/hdd/AGAC/AGAC2_models", model_serial_number)
os.makedirs(model_serial_path)

### ---|---|---|---|---|---|---|---|---|---|--- DATASET & MODEL ---|---|---|---|---|---|---|---|---|---|--- ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

dataset = import_dataset(dataset_type, root_dirs)
test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

loaded_model = import_model(model_type).to(device)
loaded_model.load_state_dict(torch.load(checkpoint_path))
loaded_model = loaded_model.eval()
### ---|---|---|---|---|---|---|---|---|---|--- TESTING ---|---|---|---|---|---|---|---|---|---|--- ###
print("testing...", flush=True)
class_correct, class_total = [0,0], [0,0]
with torch.no_grad():
    for test_data, test_labels in tqdm(test_dataloader):
        test_data, test_labels = test_data.type(torch.float).to(device), test_labels.to(device)

        test_outputs = loaded_model(test_data)
        
        test_outputs = test_outputs[:,0].type(torch.float)        
        test_outputs[test_outputs >= THRESHOLD] = 1
        test_outputs[test_outputs < THRESHOLD] = 0

        test_labels = test_labels.type(torch.int)
        correct = (test_outputs == test_labels).squeeze()
        for e,label in enumerate(test_labels):
            class_correct[label] += correct[e].item()
            class_total[label] += 1

print("Total accuracy:", sum(class_correct)/sum(class_total), " on threshold:", THRESHOLD)
print("Real detection: ", "0", "/", "0", " | accuracy:", 0, sep="")
print("Fake detection: ", class_correct[1], "/", class_total[1], " | accuracy:", class_correct[1]/class_total[1], sep="")

with open(os.path.join(model_serial_path, "test_accuracy.txt"), 'w') as txt:
    txt.write("Total accuracy: " + str(sum(class_correct)/sum(class_total)) + " on threshold: " + str(THRESHOLD) + "\n")
    txt.write("Real detection: " + "0" + "/" + "0" + " | accuracy: " + "0" + "\n")
    txt.write("Fake detection: " + str(class_correct[1]) + "/" + str(class_total[1]) + " | accuracy: " + str(class_correct[1]/class_total[1]) + "\n")

torch.cuda.empty_cache()
