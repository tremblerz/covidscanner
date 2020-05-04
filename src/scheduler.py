import torch
from torchvision import datasets, transforms
from img_datasets import CovidNetDataset
import numpy as np
from torch.autograd import Variable
import os
from utils import LoggerUtils


class Scheduler():
    """docstring for Scheduler"""

    def __init__(self, config, objects):
        super(Scheduler, self).__init__()
        self.config = config
        self.objects = objects
        self.epoch = 0
        self.initialize()

    def initialize(self):
        self.setup_data_pipeline()
        self.setup_training_params()
        log_config = {"log_path": self.config["log_path"]}
        self.logger = LoggerUtils(log_config)

    def get_split(self, dataset):

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.config["train_split"] * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices = indices[:split], indices[split:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        return train_dataset, test_dataset

    def setup_data_pipeline(self):
        self.IM_SIZE = self.config["img_size"]
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()])

        train_config = {"transforms": trainTransform,
                        "root_path": self.config["dataset_path"] + "train/",
                        "file_path": self.config["dataset_path"] + "train_split_v3.txt"}

        test_config = {"transforms": trainTransform,
                        "root_path": self.config["dataset_path"] + "test/",
                        "file_path": self.config["dataset_path"] + "test_split_v3.txt"}

        if self.config["dataset"] == "covidnet_full":
            train_dataset = CovidNetDataset(train_config)
            test_dataset = CovidNetDataset(test_config)

        if self.config["split"] is True:
            train_dataset, test_dataset = self.get_split(dataset)

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=True, num_workers=5)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5)

    def setup_training_params(self):
        self.epoch = 0
        self.model = self.objects["model"]

        self.loss_fn = self.objects["loss_fn"]
        self.optim = self.objects["optim"]
        self.device = self.objects["device"]

        self.model_path = self.config["model_path"] + "/model_pred.pt"

    def test(self):
        self.model.eval()
        test_loss, pred_correct = 0, 0
        total = 0
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.testloader):
                data = Variable(data).to(self.device)
                labels = Variable(labels).to(self.device)

                prediction = self.model(data)

                print(prediction.argmax(dim=1).sum().item())

                test_loss += self.loss_fn(prediction, labels)
                pred_correct += (prediction.argmax(dim=1) ==
                                labels).sum().item()
                total += int(data.shape[0])

        test_loss /= total
        pred_acc = pred_correct / total

        self.logger.log_scalar("test/loss", test_loss.item(), self.epoch)
        self.logger.log_scalar("test/pred_accuracy", pred_acc, self.epoch)

        self.logger.log_console("epoch {}, average test loss {:.4f}, pred_accuracy {:.3f}".format(self.epoch,
                                                              test_loss,
                                                              pred_acc))

    def train(self):
        train_loss = 0
        pred_correct_total = 0
        total = 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)
            self.optim.zero_grad()

            prediction = self.model(data)

            loss = self.loss_fn(prediction, labels)
            pred_correct = (prediction.argmax(dim=1) == labels).sum().item()

            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            pred_correct_total += pred_correct
            total += int(data.shape[0])

            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss.item(), step_num)
            self.logger.log_scalar("train/accuracy", pred_correct_total / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, accuracy {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                loss / len(data),
                                                pred_correct / len(data)))
                self.logger.log_model_stats(self.model)
                self.logger.save_model(self.model, self.model_path)

        self.logger.log_console("epoch {}, train loss {:.4f}, accuracy {:.3f}"
                                .format(self.epoch, train_loss / total,
                                        pred_correct_total / total))
        self.epoch += 1
