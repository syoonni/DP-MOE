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
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            expert_inputs = {
                name: data.to(self.device) 
                for name, data in batch['inputs'].items()
            }
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 모델 forward pass
            combined_out, gate_weights, expert_outputs, residual_out = self.model(expert_inputs)
            
            # 기본 confidence를 1로 설정 (모든 샘플에 동일한 가중치)
            confidences = torch.ones(targets.size(0), device=self.device)
            
            # Loss 계산 (MoELoss 사용)
            loss = self.criterion(combined_out, expert_outputs, gate_weights, targets, confidences)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy 계산
            _, predicted = combined_out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:  # 진행상황 출력
                print(f'Batch [{batch_idx}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}, '
                    f'Acc: {100. * correct/total:.2f}%')
        
        return total_loss / len(train_loader), correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        all_gate_weights = []
        
        with torch.no_grad():
            for batch in val_loader:
                expert_inputs = {
                    name: data.to(self.device) 
                    for name, data in batch['inputs'].items()
                }
                targets = batch['targets'].to(self.device)
                
                # 모델 forward pass
                combined_out, gate_weights, expert_outputs, residual_out = self.model(expert_inputs)
                
                # 기본 confidence를 1로 설정
                confidences = torch.ones(targets.size(0), device=self.device)
                
                # Loss 계산
                loss = self.criterion(combined_out, expert_outputs, gate_weights, targets, confidences)
                
                total_loss += loss.item()
                
                _, predicted = combined_out.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu().numpy())
        
        return (
            total_loss / len(val_loader),
            correct / total,
            np.array(all_predictions),
            np.array(all_targets),
            np.concatenate(all_gate_weights, axis=0)
        )
    

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
            
            # Validate와 반환값 처리 수정
            val_metrics = self.validate(val_loader)
            val_loss, val_acc = val_metrics[:2]  # 앞의 두 값만 사용

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
                batch = next(iter(train_loader))
                expert_inputs = batch['inputs']
                expert_inputs = {
                    name: data.to(self.device) 
                    for name, data in expert_inputs.items()
                }
                _, gate_weights, _, _ = self.model(expert_inputs)
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

        return self.best_loss, val_acc
            