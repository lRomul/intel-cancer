import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scripts.model import Model


def get_pretrained_model(arch, lr=1e-4, momentum=0.9, weight_decay=1e-4):
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise Exception("Not supported architecture: %s"%arch)
    
    if arch.startswith('vgg'):
        mod = list(model.classifier.children())
        mod.pop()
        mod.append(torch.nn.Linear(4096, 3))
        new_classifier = torch.nn.Sequential(*mod)
        model.classifier = new_classifier
    elif arch.startswith('resnet'):
        model.fc = nn.Linear(2048, 3)
    else:
        raise Exception("Not supported architecture: %s"%arch)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    
    return Model(model, criterion, optimizer)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        s = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.input_conv = BasicConv2d(3, s, 1)
        self.conv_1 = BasicConv2d(s * 1, s * 1, 3, padding=1)
        self.conv_2 = BasicConv2d(s * 1, s * 1, 3, padding=1)
        self.conv_3 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.conv_4 = BasicConv2d(s * 2, s * 2, 3, padding=1)
        self.conv_5 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.conv_6 = BasicConv2d(s * 4, s * 4, 3, padding=1)
        self.fc_1 = nn.Linear(2304, 256)
        self.fc_2 = nn.Linear(256, 4)
        self.dropout2d = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.input_conv(x)
        
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = self.conv_3(x)
        x = self.pool(x)
        x = self.conv_4(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = self.conv_5(x)
        x = self.pool(x)
        x = self.conv_6(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc_2(x)
        x = self.relu(x)
        
        return x
    