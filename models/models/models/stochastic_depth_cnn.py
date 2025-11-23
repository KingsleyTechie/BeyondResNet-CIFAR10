import torch
import torch.nn as nn
import random

class StochasticDepthCNN(nn.Module):
    def __init__(self, config):
        super(StochasticDepthCNN, self).__init__()
        self.config = config
        self.stochastic_depth_prob = config['stochastic_depth_prob']
        
        # Feature extractor with attention
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout_rate']),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout_rate']),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout_rate']),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256 // config['reduction_ratio']),
            nn.ReLU(inplace=True),
            nn.Linear(256 // config['reduction_ratio'], 256),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(config['classifier_dropout']),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        
        # Apply attention
        attention_weights = self.attention(x).view(-1, 256, 1, 1)
        x = x * attention_weights
        
        # Stochastic depth: randomly skip classifier during training
        if self.training and random.random() < self.stochastic_depth_prob:
            # Use global average pooling as fallback
            x = x.mean(dim=[2, 3])
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
        return x
