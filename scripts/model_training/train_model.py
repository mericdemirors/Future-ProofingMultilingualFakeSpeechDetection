print("imports...", flush=True)
import os
import argparse
import glob

import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore")

from datasets import *
from models import *

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-model_type', '-m', type=str, default="mid_CNN", help='model type to train. default is: \"mid_CNN\"')
    parser.add_argument('-root_dirs', '-r', type=str, default="", nargs='*', help='list of image dataset folders. default is: \"\"')
    parser.add_argument('-root_dirs_code', '-c', type=str, default="", help='code name for current train dataset. default is: \"\"')
    parser.add_argument('-dataset_type', '-d', type=str, default="gray", help='Dataset type to train on. default is: \"gray\"')
    parser.add_argument('-split', '-s', type=float, nargs=2, default=(0.8,0.1), help='Train validation split as two floats separated by space(remaining will be test). default is: (0.8,0.1)')
    parser.add_argument('-epochs', '-e', type=int, default=100, help='Number of epochs. default is: 100')
    parser.add_argument('-batch', '-b', type=int, default=32, help='Batch size. default is: 32')
    parser.add_argument('-LR', '-lr', type=float, default=0.001, help='Learning rate. default is: 0.001')
    parser.add_argument('-threshold', '-th', type=float, default=0.5, help='Threshold for sigmoid classification. default is: 0.5')
    return parser.parse_args()

args = parse_arguments()
model_type = args.model_type
root_dirs = args.root_dirs
root_dirs_code = args.root_dirs_code
dataset_type = args.dataset_type
TRAIN_VALIDATION_SPLIT = tuple(args.split)
NUM_OF_EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.LR
THRESHOLD = args.threshold

model_serial_number = f"{model_type}_{os.path.split(root_dirs_code)[1]}_{datetime.now().strftime('%m:%d:%H:%M:%S')}"
print("serial_number:", model_serial_number, flush=True)
model_serial_path = os.path.join("/hdd/AGAC/AGAC2_models", model_serial_number)
os.makedirs(model_serial_path)

### ---|---|---|---|---|---|---|---|---|---|--- DATASET & MODEL ---|---|---|---|---|---|---|---|---|---|--- ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

dataset = import_dataset(dataset_type, root_dirs)

train_size = int(TRAIN_VALIDATION_SPLIT[0] * len(dataset))
validation_size = int(TRAIN_VALIDATION_SPLIT[1] * len(dataset))
test_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = import_model(model_type).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

### ---|---|---|---|---|---|---|---|---|---|--- TRAINING ---|---|---|---|---|---|---|---|---|---|--- ###
print("training...", flush=True)
torch.cuda.empty_cache()

train_losses, validation_losses = [], []
min_validation_loss = None
old_min_validation_save = None
early_stop_step = 0

for epoch in range(NUM_OF_EPOCHS):
    torch.cuda.empty_cache()
    for batch_idx, (data, labels) in tqdm(enumerate(train_dataloader), leave=False):
        data, labels = data.to(device), labels.to(device)

        model = model.train()
        outputs = model(data)

        float_outputs = outputs[:,0].type(torch.float)
        float_labels = labels.type(torch.float)
        train_loss = criterion(float_outputs, float_labels)
        
        optimizer.zero_grad()
        train_loss.backward()    
        optimizer.step()

    with torch.no_grad():
        model = model.eval()
        validation_loss = 0.0
        for validation_data, validation_labels in validation_dataloader:
            validation_data, validation_labels = validation_data.to(device), validation_labels.to(device)

            validation_outputs = model(validation_data)
            validation_loss += criterion(validation_outputs[:,0].type(torch.float), validation_labels.type(torch.float)).item()
        validation_loss /= len(validation_dataset)
    
    if min_validation_loss is None or validation_loss < min_validation_loss:
        early_stop_step = 0
        min_validation_loss = validation_loss
        if old_min_validation_save is not None:
            os.remove(os.path.join(model_serial_path, old_min_validation_save))
        save_name = "min_validation_loss:"+str(min_validation_loss) + "_epoch:" + str(epoch) + ".pth"
        old_min_validation_save = save_name
        torch.save(model.state_dict(), os.path.join(model_serial_path, save_name))

    print("Epoch: ", epoch+1, " | training loss: ", train_loss.item(), " | min validation loss: ", min_validation_loss, sep="", flush=True)
    train_losses.append(train_loss.item())
    validation_losses.append(validation_loss)
    scheduler.step()
    
    early_stop_step = early_stop_step + 1
    if early_stop_step == 10:
        print("early stopping...")
        break

plt.plot(train_losses, color='blue', label='Train Loss')
plt.plot(validation_losses, color='orange', label='Validation Loss')
plt.legend()
plt.savefig(os.path.join(model_serial_path, "losses.png"))

### ---|---|---|---|---|---|---|---|---|---|--- TESTING ---|---|---|---|---|---|---|---|---|---|--- ###
print("testing...", flush=True)
loaded_model = import_model(model_type).to(device)
loaded_model.load_state_dict(torch.load(os.path.join(model_serial_path, old_min_validation_save)))
loaded_model = loaded_model.eval()

class_correct, class_total = [0,0], [0,0]
with torch.no_grad():
    for test_data, test_labels in test_dataloader:
        test_data, test_labels = test_data.to(device), test_labels.to(device)

        test_outputs = loaded_model(test_data)
        
        test_outputs = test_outputs[:,0].type(torch.float)
        test_outputs[test_outputs >= THRESHOLD] = 1
        test_outputs[test_outputs < THRESHOLD] = 0

        correct = (test_outputs == test_labels).squeeze()
        for e,label in enumerate(test_labels):
            class_correct[label] += correct[e].item()
            class_total[label] += 1

print("Total accuracy:", sum(class_correct)/sum(class_total), " on threshold:", THRESHOLD)
print("Real detection: ", class_correct[0], "/", class_total[0], " | accuracy:", class_correct[0]/class_total[0], sep="")
print("Fake detection: ", class_correct[1], "/", class_total[1], " | accuracy:", class_correct[1]/class_total[1], sep="")

with open(os.path.join(model_serial_path, "accuracy.txt"), 'w') as txt:
    txt.write("Total accuracy: " + str(sum(class_correct)/sum(class_total)) + " on threshold: " + str(THRESHOLD) + "\n")
    txt.write("Real detection: " + str(class_correct[0]) + "/" + str(class_total[0]) + " | accuracy: " + str(class_correct[0]/class_total[0]) + "\n")
    txt.write("Fake detection: " + str(class_correct[1]) + "/" + str(class_total[1]) + " | accuracy: " + str(class_correct[1]/class_total[1]) + "\n")

torch.cuda.empty_cache()
