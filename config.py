import torch

class Config:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data configuration
    dataset = 'CIFAR10'
    data_path = './data'
    batch_size = 128
    test_batch_size = 100
    num_workers = 2
    
    # Training configuration
    epochs = 25
    learning_rate = 0.001
    weight_decay = 1e-4
    scheduler_step = 15
    scheduler_gamma = 0.1
    
    # Model configurations
    baseline_config = {
        'conv_channels': [64, 128, 256],
        'dropout_rate': 0.25,
        'classifier_dropout': 0.5
    }
    
    attention_config = {
        'conv_channels': [64, 128, 256],
        'dropout_rate': 0.25,
        'classifier_dropout': 0.5,
        'reduction_ratio': 16
    }
    
    stochastic_depth_config = {
        'conv_channels': [64, 128, 256],
        'dropout_rate': 0.25,
        'classifier_dropout': 0.5,
        'reduction_ratio': 16,
        'stochastic_depth_prob': 0.2
    }
