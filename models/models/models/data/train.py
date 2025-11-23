import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
from datetime import datetime

from models.baseline_cnn import BaselineCNN
from models.attention_cnn import AttentionCNN
from models.stochastic_depth_cnn import StochasticDepthCNN
from data.data_loader import get_data_loaders
from utils.metrics import calculate_accuracy, calculate_classification_report
from utils.visualization import plot_training_history

class Trainer:
    def __init__(self, config, model_type='baseline'):
        self.config = config
        self.model_type = model_type
        self.device = config.device
        
        # Create model
        if model_type == 'baseline':
            self.model = BaselineCNN(config.baseline_config)
        elif model_type == 'attention':
            self.model = AttentionCNN(config.attention_config)
        elif model_type == 'stochastic_depth':
            self.model = StochasticDepthCNN(config.stochastic_depth_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self.train_loader, self.test_loader = get_data_loaders(
            config.data_path,
            config.batch_size,
            config.test_batch_size,
            config.num_workers
        )
        
        # Setup optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step,
            gamma=config.scheduler_gamma
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'learning_rates': []
        }
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, epochs=None):
        if epochs is None:
            epochs = self.config.epochs
        
        print(f"Training {self.model_type} model for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch()
            
            # Evaluation phase
            test_accuracy = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['test_accuracy'].append(test_accuracy)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Loss: {train_loss:.4f}, '
                  f'Accuracy: {test_accuracy:.2f}%, '
                  f'LR: {current_lr:.6f}')
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self.save_checkpoint(epoch, best_accuracy)
                
                if test_accuracy > 90.0:
                    print(f"Breakthrough: Achieved >90% accuracy! ({test_accuracy:.2f}%)")
        
        # Final evaluation and reporting
        final_report = calculate_classification_report(self.model, self.test_loader, self.device)
        self.save_results(final_report, best_accuracy)
        
        return best_accuracy
    
    def save_checkpoint(self, epoch, accuracy):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/{self.model_type}_cnn.pth')
    
    def save_results(self, report, best_accuracy):
        results = {
            'model_type': self.model_type,
            'best_accuracy': best_accuracy,
            'final_accuracy': self.history['test_accuracy'][-1],
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'training_history': self.history,
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{self.model_type}', exist_ok=True)
        
        with open(f'results/{self.model_type}/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training history
        plot_training_history(self.history, self.model_type)

def main():
    from config import Config
    
    config = Config()
    
    # Train baseline model
    print("=== Training Baseline CNN ===")
    baseline_trainer = Trainer(config, 'baseline')
    baseline_accuracy = baseline_trainer.train()
    
    # Train attention model
    print("\n=== Training Attention CNN ===")
    attention_trainer = Trainer(config, 'attention')
    attention_accuracy = attention_trainer.train()
    
    # Train stochastic depth model
    print("\n=== Training Stochastic Depth CNN ===")
    sd_trainer = Trainer(config, 'stochastic_depth')
    sd_accuracy = sd_trainer.train()
    
    # Print final comparison
    print("\n=== Final Results ===")
    print(f"Baseline CNN: {baseline_accuracy:.2f}%")
    print(f"Attention CNN: {attention_accuracy:.2f}%")
    print(f"Stochastic Depth CNN: {sd_accuracy:.2f}%")

if __name__ == "__main__":
    main()
