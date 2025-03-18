import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import os
from sklearn.metrics import roc_curve, auc
from data.data_processor import Dataprocessor
from evaluation.evaluator import MoEEvaluator
from training.MoELoss import MoELoss

class MoETrainer:
    def __init__(self, model, optimizer, scheduler, device, expert_features, alpha=0.1, beta=0.1, class_weights=None):
        self.model = model
        self.criterion = MoELoss(alpha=alpha, beta=beta, class_weights=class_weights)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.expert_features = expert_features
        self.best_loss = float('inf')
        # Initialize evaluator
        self.evaluator = MoEEvaluator(model, device, expert_features)

    def calculate_accuracy(self, data_loader):
        """Calculate accuracy for the given data loader"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target, _ in data_loader:
                expert_inputs = {}
                start_idx = 0
                for name, features in self.expert_features.items():
                    num_features = len(features)
                    expert_inputs[name] = data[:, start_idx:start_idx+num_features].to(self.device)
                    start_idx += num_features

                target = target.to(self.device, dtype=torch.int64)
                combined_out, _, _ = self.model(expert_inputs)
                
                _, predicted = torch.max(combined_out.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for data, target, _ in train_loader:
            expert_inputs = {}
            start_idx = 0
            for name, features in self.expert_features.items():
                num_features = len(features)
                expert_inputs[name] = data[:, start_idx:start_idx+num_features].to(self.device)
                start_idx += num_features
            
            target = target.to(self.device, dtype=torch.int64)

            self.optimizer.zero_grad()

            for name in expert_inputs:
                expert_inputs[name] = Dataprocessor.add_noise(expert_inputs[name])
            
            combined_out, gate_weights, expert_outputs = self.model(expert_inputs)

            confidences = torch.max(torch.softmax(combined_out, dim=1), dim=1)[0]

            loss = self.criterion(combined_out, expert_outputs, gate_weights, target, confidences)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        train_acc = self.calculate_accuracy(train_loader)
        return running_loss / len(train_loader), train_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data, target, _ in val_loader:
                expert_inputs = {}
                start_idx = 0
                for name, features in self.expert_features.items():
                    num_features = len(features)
                    expert_inputs[name] = data[:, start_idx:start_idx+num_features].to(self.device)
                    start_idx += num_features

                target = target.to(self.device, dtype=torch.int64)

                combined_out, gate_weights, expert_outputs = self.model(expert_inputs)

                confidences = torch.max(torch.softmax(combined_out, dim=1), dim=1)[0]

                loss = self.criterion(combined_out, expert_outputs, gate_weights, target, confidences)

                running_loss += loss.item()

        val_acc = self.calculate_accuracy(val_loader)
        return running_loss / len(val_loader), val_acc
    

    def train(self, train_loader, val_loader, num_epochs, best_model_save_path, 
              final_model_save_path, output_dir):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        gate_weights_history = []

        # Create metrics DataFrame
        metrics_data = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Store metrics for CSV
            metrics_data['epoch'].append(epoch + 1)
            metrics_data['train_loss'].append(train_loss)
            metrics_data['train_acc'].append(train_acc)
            metrics_data['val_loss'].append(val_loss)
            metrics_data['val_acc'].append(val_acc)

            self.evaluator.log_accuracy(train_acc, val_acc)

            # Get average gate weights for this epoch
            self.model.eval()
            with torch.no_grad():
                data, _, _ = next(iter(train_loader))
                expert_inputs = {}
                start_idx = 0
                for name, features in self.expert_features.items():
                    num_features = len(features)
                    expert_inputs[name] = data[:, start_idx:start_idx+num_features].to(self.device)
                    start_idx += num_features
                _, gate_weights, _ = self.model(expert_inputs)
                gate_weights_history.append(gate_weights.mean(dim=0).cpu().numpy())
            self.model.train()

            # Update learning rate
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)


            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_state = {
                    'State_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                    'scheduler': copy.deepcopy(self.scheduler.state_dict()),
                }
                torch.save(best_state, best_model_save_path)

            # Save final model
            final_state = {
                'State_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'scheduler': copy.deepcopy(self.scheduler.state_dict()),
            }
            torch.save(final_state, final_model_save_path)

            # Plot training metrics
            plt.figure(figsize=(15, 5))
            
            # Plot losses
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot accuracies
            plt.subplot(1, 3, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(val_accs, label='Validation Accuracy')
            plt.title('Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot gate weights
            if len(gate_weights_history) > 0:
                plt.subplot(1, 3, 3)
                weights = np.array(gate_weights_history)
                plt.plot(weights[:, 0], label='Expert 1 Weight')
                plt.plot(weights[:, 1], label='Expert 2 Weight')
                plt.title('Gate Weights Evolution')
                plt.xlabel('Epoch')
                plt.ylabel('Weight')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_plots.png'))
            plt.close()
            