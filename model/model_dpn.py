import torch
import torch.nn as nn
import torchvision
from torchvision import models

class Resnet34Fc(nn.Module):
  def __init__(self):
    super(Resnet34Fc, self).__init__()
    model_resnet34 = models.resnet34(pretrained=True)
    self.conv1 = model_resnet34.conv1
    self.bn1 = model_resnet34.bn1
    self.relu = model_resnet34.relu
    self.maxpool = model_resnet34.maxpool
    self.layer1 = model_resnet34.layer1
    self.layer2 = model_resnet34.layer2
    self.layer3 = model_resnet34.layer3
    self.layer4 = model_resnet34.layer4
    self.avgpool = model_resnet34.avgpool
    self.__in_features = model_resnet34.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5, classaware_dp=False):
        super(DomainPredictor, self).__init__()
        self.dp_model = Resnet34Fc()
        self.classaware_dp=classaware_dp
        for param in self.dp_model.conv1.parameters():
            param.requires_grad = False
        for param in self.dp_model.bn1.parameters():
            param.requires_grad = False
        for param in self.dp_model.layer1.parameters():
            param.requires_grad = True
        for param in self.dp_model.layer2.parameters():
            param.requires_grad = True
        self.inp_layer = 2048 if classaware_dp else 2048
        self.fc5 = nn.Linear(self.inp_layer, 128)
        self.bn_fc5 = nn.BatchNorm1d(128)
        self.dp_layer = nn.Linear(128, num_domains)

        self.prob = prob
        self.num_domains = num_domains

        self.relu = nn.ReLU(inplace=True)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if not self.classaware_dp:
            x = self.dp_model.conv1(x)
            x = self.dp_model.bn1(x)
            x = self.dp_model.relu(x)
            x = self.dp_model.maxpool(x)

            x = self.dp_model.layer1(x)
            x = self.dp_model.layer2(x)

            x = self.dp_model.layer3(x)
            x = self.dp_model.layer4(x)
            x = self.dp_model.avgpool(x)
            x = x.view(x.size(0), -1)

        x = self.relu(self.bn_fc5(self.fc5(x)))

        dp_pred = self.dp_layer(x)
        return dp_pred
