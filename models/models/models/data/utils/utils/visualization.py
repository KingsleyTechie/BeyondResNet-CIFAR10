import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(history, model_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(history['train_loss'])
    ax1.set_title(f'{model_type} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax2.plot(history['test_accuracy'])
    ax2.set_title(f'{model_type} - Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_type}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
