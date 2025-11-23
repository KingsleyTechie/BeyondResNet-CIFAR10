import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, config):
        super(BaselineCNN, self).__init__()
        self.config = config
        
        # Feature extractor
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout_rate']),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout_rate']),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(config['dropout_rate']),
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    config = {
        'conv_channels': [64, 128, 256],
        'dropout_rate': 0.25,
        'classifier_dropout': 0.5
    }
    model = BaselineCNN(config)
    print(f"Baseline CNN parameters: {count_parameters(model):,}")
