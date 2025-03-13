import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class PretrainedResNet(nn.Module):
    def __init__(self):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        # freezing parameters before (layer2)
        for i,p in enumerate(self.model.parameters()):
            if i < 33:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
                
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)

class ResNetGray(nn.Module):
    def __init__(self):
        super(ResNetGray, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class ResNetRGB(nn.Module):
    def __init__(self):
        super(ResNetRGB, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class ResNetMulti(nn.Module):
    def __init__(self):
        super(ResNetMulti, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
        self.model.conv1 = nn.Conv2d(5, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class PretrainedGoogleNet(nn.Module):
    def __init__(self):
        super(PretrainedGoogleNet, self).__init__()
        self.model = models.googlenet(pretrained=True)

        # freezing parameters before inception4a
        for i,p in enumerate(self.model.parameters()):
            if i < 45:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

class GoogleNetGray(nn.Module):
    def __init__(self):
        super(GoogleNetGray, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
        self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class GoogleNetRGB(nn.Module):
    def __init__(self):
        super(GoogleNetRGB, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class GoogleNetMulti(nn.Module):
    def __init__(self):
        super(GoogleNetMulti, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
        self.model.conv1.conv = nn.Conv2d(5, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class MidCNN(nn.Module):
    def __init__(self):
        super(MidCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3)
        self.drop5 = nn.Dropout2d(p=0.2)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=3)
        self.drop6 = nn.Dropout2d(p=0.2)
        self.relu = F.relu
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.sigmoid = nn.Sigmoid()

        self.fully_conv = nn.Conv2d(16, 1, kernel_size=6)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(self.relu(self.bn2(self.conv2(x))))
        x = self.max_pool(self.relu(self.bn3(self.conv3(x))))
        
        x = self.max_pool(self.relu(self.bn4(self.conv4(x))))

        x = self.drop5(self.relu(self.conv5(x)))
        x = self.drop6(self.relu(self.conv6(x)))
        x = self.sigmoid(self.fully_conv(x))
        
        return x[:,0,0]

class MidCNNMulti(nn.Module):
    def __init__(self):
        super(MidCNNMulti, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3)
        self.drop5 = nn.Dropout2d(p=0.2)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=3)
        self.drop6 = nn.Dropout2d(p=0.2)
        self.relu = F.relu
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.sigmoid = nn.Sigmoid()

        self.fully_conv = nn.Conv2d(16, 1, kernel_size=6)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(self.relu(self.bn2(self.conv2(x))))
        x = self.max_pool(self.relu(self.bn3(self.conv3(x))))
        
        x = self.max_pool(self.relu(self.bn4(self.conv4(x))))

        x = self.drop5(self.relu(self.conv5(x)))
        x = self.drop6(self.relu(self.conv6(x)))
        x = self.sigmoid(self.fully_conv(x))
        
        return x[:,0,0]

class PretrainedAlexNet(nn.Module):
    def __init__(self):
        super(PretrainedAlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        
        # freezing parameters before (layer2)
        for layer in self.model.features:
            layer.requires_grad_(False)
                
        self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

class AlexNetGray(nn.Module):
    def __init__(self):
        super(AlexNetGray, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return nn.Sigmoid()(x)

class AlexNetGray_eski(nn.Module):
    def __init__(self):
        super(AlexNetGray_eski, self).__init__()
        self.model = models.alexnet(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=True)

        self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = nn.ReLU()(self.model(x))
        x = self.fc(x)
        return x

class AlexNetRGB(nn.Module):
    def __init__(self):
        super(AlexNetRGB, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return nn.Sigmoid()(x)

class AlexNetRGB_eski(nn.Module):
    def __init__(self):
        super(AlexNetRGB_eski, self).__init__()
        self.model = models.alexnet(pretrained=False)                
        self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

class AlexNetMulti(nn.Module):
    def __init__(self):
        super(AlexNetMulti, self).__init__()
        self.model = models.alexnet(pretrained=False, num_classes=1)
        self.model.features[0] = nn.Conv2d(5, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class PretrainedVGG16(nn.Module):
    def __init__(self):
        super(PretrainedVGG16, self).__init__()
        self.model = models.vgg16_bn(pretrained=True)
        
        # freezing parameters before (layer2)
        for layer in self.model.features:
            layer.requires_grad_(False)
                
        self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

class VGG16Gray(nn.Module):
    def __init__(self):
        super(VGG16Gray, self).__init__()
        self.model = models.vgg16_bn(pretrained=False, num_classes=1)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class VGG16RGB(nn.Module):
    def __init__(self):
        super(VGG16RGB, self).__init__()
        self.model = models.vgg16_bn(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class VGG16Multi(nn.Module):
    def __init__(self):
        super(VGG16Multi, self).__init__()
        self.model = models.vgg16_bn(pretrained=False, num_classes=1)
        self.model.features[0] = nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))


def import_model(model_type):
    if model_type == "pt_resnet":
        return PretrainedResNet()
    elif model_type == "resnet_gray":
        return ResNetGray()
    elif model_type == "resnet_RGB":
        return ResNetRGB()
    elif model_type == "resnet_multi":
        return ResNetMulti()
    
    elif model_type == "pt_google":
        return PretrainedGoogleNet()
    elif model_type == "google_gray":
        return GoogleNetGray()
    elif model_type == "google_RGB":
        return GoogleNetRGB()
    elif model_type == "google_multi":
        return GoogleNetMulti()
    
    elif model_type == "pt_alexnet":
        return PretrainedAlexNet()
    elif model_type == "alexnet_gray":
        return AlexNetGray()
    elif model_type == "alexnet_RGB":
        return AlexNetRGB()
    elif model_type == "alexnet_multi":
        return AlexNetMulti()
    
    elif model_type == "pt_vgg":
        return PretrainedVGG16()
    elif model_type == "vgg_gray":
        return VGG16Gray()
    elif model_type == "vgg_RGB":
        return VGG16RGB()
    elif model_type == "vgg_multi":
        return VGG16Multi()

    elif model_type == "mid_CNN":
        return MidCNN()
    elif model_type == "mid_CNN_multi":
        return MidCNNMulti()
