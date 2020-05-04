from torchvision import models
import torch.nn as nn

distance = nn.CrossEntropyLoss()


def loss_fn(logits, labels):
    CE = distance(logits, labels)
    return CE


class resnet18(nn.Module):
    """docstring for ResNet"""

    def __init__(self, config):
        super(resnet18, self).__init__()
        self.logits = config["logits"]

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
        return nn.functional.softmax(x)
