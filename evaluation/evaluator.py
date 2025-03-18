import torch
import pandas as pd
import numpy as np
import torchmetrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MoEEvaluator:
    def __init__(self, model, device, expert_features):
        self.model = model
        self.device = device
        self.expert_features = expert_features
        self.metrics = self._initialize_metrics()
        self.train_accuracies = []
        self.val_accuracies = []

    def _initialize_metrics(self):
        def create_metric(metric_class, **kwargs):
            return metric_class(task='binary', **kwargs).to(self.device)
        
        return {
            'accuracy': create_metric(torchmetrics.Accuracy),
            'precision': create_metric(torchmetrics.Precision),
            'recall': create_metric(torchmetrics.Recall),
            'specificity': create_metric(torchmetrics.Specificity),
            'f1': create_metric(torchmetrics.F1Score),
            'auroc': create_metric(torchmetrics.AUROC)
        }
    
    def log_accuracy(self, train_acc, val_acc):
        """Log accuracy values for plotting"""
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)


    def evaluate(self, test_loader, model_path, output_dir):
        # Load model state
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['State_dict'])
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_gate_weights = []
        all_expert_outputs = {} # 딕셔너리로 초기화
        all_ids = []

        with torch.no_grad():
            for data, target, ids in test_loader:
                expert_inputs = {}
                start_idx = 0
                for name, features in self.expert_features.items():
                    num_features = len(features)
                    expert_inputs[name] = data[:, start_idx:start_idx+num_features].to(self.device)
                    start_idx += num_features

                target = target.to(self.device, dtype=torch.int64)

                combined_out, gate_weights, expert_outputs = self.model(expert_inputs)
                
                all_predictions.append(combined_out)
                all_targets.append(target)
                all_gate_weights.append(gate_weights)

                for name, output in expert_outputs.items():
                    if name not in all_expert_outputs:
                        all_expert_outputs[name] = []
                    all_expert_outputs[name].append(output)

                all_ids.extend(ids.numpy())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        gate_weights = torch.cat(all_gate_weights, dim=0)

        for name in all_expert_outputs:
            all_expert_outputs[name] = torch.cat(all_expert_outputs[name], dim=0)

        results = self.calculate_and_save_metrics(
            predictions, targets, gate_weights, all_expert_outputs, output_dir, all_ids
        )

        return results

    def calculate_and_save_metrics(self, predictions, targets, gate_weights, expert_outputs, output_dir, all_ids):
        results = {}
        predictions_class = torch.argmax(predictions, dim=1)

        # Get probabilities for positive class 
        probabilities = torch.softmax(predictions, dim=1)[:, 1].cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate metrics
        for name, metric in self.metrics.items():
            if name == 'auroc':
                scores = torch.softmax(predictions, dim=1)[:, 1]  # positive class의 확률값
                scores = scores.cpu().float()
                target_cpu = targets.cpu().float()
                
                metric.update(scores, target_cpu)
                results[name] = metric.compute().item()
            elif name == 'recall':
                # Manual recall calculation from confusion matrix
                cm = confusion_matrix(targets.cpu().numpy(), predictions_class.cpu().numpy())
                tn, fp, fn, tp = cm.ravel()
                results[name] = tp / (tp + fn)  # Manual recall calculation
            else:
                metric.update(predictions_class, targets)
                results[name] = metric.compute().item()

        # Plot and save ROC curve
        fpr, tpr, _ = roc_curve(targets_np, probabilities)
        roc_auc = auc(fpr, tpr)

        roc_df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': _
        })
        roc_df.to_csv(os.path.join(output_dir, 'roc_data.csv'), index=False)

        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

        # Save confusion matrix
        cm = confusion_matrix(
            targets.cpu().numpy(),
            predictions_class.cpu().numpy(),
            normalize='true'
        )
        tn, fp, fn, tp = cm.ravel()

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - MoE Model')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # Plot gate weights distribution
        plt.figure(figsize=(10, 5))
        gate_weights_np = gate_weights.cpu().numpy()
        for i in range(gate_weights_np.shape[1]):
            plt.hist(gate_weights_np[:, i], bins=30, alpha=0.5, 
                    label=f'Expert {i+1}')
        plt.title('Gate Weights Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'gate_weights_dist.png'))
        plt.close()

        # Save detailed predictions and analysis
        predictions_df = pd.DataFrame({
            'id': all_ids,
            'True_Label': targets.cpu().numpy(),
            'Predicted_Label': predictions_class.cpu().numpy(),
            'Prob_Class_0': predictions[:, 0].cpu().numpy(),
            'Prob_Class_1': predictions[:, 1].cpu().numpy(),
        })

        # Add gate weights for each expert
        for i in range(gate_weights.shape[1]):
            predictions_df[f'Expert{i+1}_Weight'] = gate_weights[:, i].cpu().numpy()

        # Add expert predictions
        for name, output in expert_outputs.items():
            expert_preds = torch.argmax(output, dim=1).cpu().numpy()
            predictions_df[f'{name}_Prediction'] = expert_preds

        predictions_df.to_csv(os.path.join(output_dir, 'predictions_moe.csv'), index=False)

        # Save expert agreement analysis
        incorrect_predictions = predictions_df[predictions_df['True_Label'] != predictions_df['Predicted_Label']]
        with open(os.path.join(output_dir, 'error_analysis.txt'), 'w') as f:
            f.write(f"Total incorrect predictions: {len(incorrect_predictions)}\n\n")
            f.write("Expert Weight Statistics for Incorrect Predictions:\n")
            for i in range(gate_weights.shape[1]):
                mean_weight = incorrect_predictions[f'Expert{i+1}_Weight'].mean()
                std_weight = incorrect_predictions[f'Expert{i+1}_Weight'].std()
                f.write(f"Expert {i+1}: {mean_weight:.4f} ± {std_weight:.4f}\n")

        return results
