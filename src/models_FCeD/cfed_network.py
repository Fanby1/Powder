import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models_Cprompt.vision_transformer import VisionTransformer
import numpy as np
import math


class cfed_network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(cfed_network, self).__init__()
        self.feature = feature_extractor
        #self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.fc = nn.Linear(768, numclass, bias=True)

    def forward(self, input):
        #x = self.feature(input)
        x, _, _ = self.feature(input)
        x = x[:,0,:]
        x = self.fc(x)
        return x
    
    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        feature, _, _ = self.feature(inputs)
        return feature[:,0,:]
    
    