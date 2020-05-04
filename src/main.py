import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import shutil

from scheduler import Scheduler
from models import resnet18, loss_fn
from utils import copy_source_code

gpu_id = 1

if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(gpu_id))
else:
    device = torch.device('cpu')

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

counter = 1
model_name = "resnet18_pretrained"
total_epochs = 300
gpu_devices = [1, 2, 3, 4, 5, 6, 7]
num_gpus = len(gpu_devices)
dataset = "covidnet_full"
dataset_path = "../data/"
img_size = 256
experiment_name = "{}_{}_{}_{}".format(dataset, img_size, model_name, counter)
results_path = "../experiments/" + experiment_name
log_path = results_path + "/logs/"
model_path = results_path + "/saved_models"
train_batch_size = 64 * num_gpus
test_batch_size = 64
split = False

config = {"experiment_name": experiment_name, "dataset_path": dataset_path,
          "total_epochs": total_epochs, "train_batch_size": train_batch_size,
          "dataset": dataset, "test_batch_size": test_batch_size, "log_path": log_path,
          "model_path": model_path, "img_size": img_size, "split": split}

hparams = {"logits": 2, "pretrained": True}

if os.path.isdir(results_path):
    print("Experiment {} already present".format(experiment_name))
    inp = input("Press e to exit, r to replace it: ")
    if inp == "e":
        exit()
    elif inp == "r":
        shutil.rmtree(results_path)
    else:
        print("Input not understood")
        exit()

copy_source_code(results_path)
os.mkdir(model_path)
os.mkdir(log_path)

def main():
    model = resnet18(hparams)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    multi_gpu_model = nn.DataParallel(model, device_ids=gpu_devices)
    objects = {"model": multi_gpu_model, "optim": optimizer, "device": gpu_id,
               "loss_fn": loss_fn}

    scheduler = Scheduler(config, objects)

    # call the training loop
    for epoch in range(config["total_epochs"]):
        scheduler.train()
        scheduler.test()

if __name__ == '__main__':
    main()
