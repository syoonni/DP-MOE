import os
import sys

# Add parent directory to python path for imports - MUST BE BEFORE LOCAL IMPORTS
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Now we can import local modules
from data.dataset import AlzheimerDataset

class ModelComparison:
    """Framework for comparing MoE with traditional ML models"""
    
    def __init__(self, data_path, all_features, labels=['MCI', 'AD'], sex='A'):
        """
        Initialize the comparison framework
        
        Args:
            data_path: Path to the dataset
            all_features: List of features to use
            labels: List of class labels
            sex: Sex filter ('A' for all)
        """
        self.data_path = data_path
        self.all_features = all_features
        self.labels = labels
        self.sex = sex
        
        # Load and preprocess data
        self.dataset = AlzheimerDataset(
            data_path=data_path,
            interesting_features=all_features,
            labels=labels,
            sex=sex
        )
        
        self.data_splits = self.dataset.load_and_split_data()
        
        # Convert data splits to numpy arrays
        self.X_train = self.data_splits['train'][0]
        self.y_train = self.data_splits['train'][1]
        self.X_val = self.data_splits['val'][0]
        self.y_val = self.data_splits['val'][1]
        self.X_test = self.data_splits['test'][0]
        self.y_test = self.data_splits['test'][1]
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Storage for results
        self.results = {}

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate model performance using multiple metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0)
        }
        
        if y_pred_proba is not None:
            metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)
            
        return metrics

    def train_svm(self, output_dir):
        """
        Train and evaluate SVM model
        """
        print("\nTraining SVM model...")
        
        # Train SVM with probability estimates and balanced class weights
        model = SVC(
            probability=True,
            random_state=42,
            class_weight='balanced',
            kernel='rbf',
            cache_size=1000  # Increase cache size for better performance
        )
        
        print("Fitting SVM model...")
        model.fit(self.X_train_scaled, self.y_train)
        
        print("Making predictions...")
        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
    
        # Calculate ROC curve data
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        
        # Save ROC data to CSV
        roc_data = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        })
        roc_data.to_csv(os.path.join(output_dir, 'roc_data.csv'), index=False)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.2f})')
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
        
        # Evaluate
        print("Evaluating model...")
        metrics = self.evaluate_model(self.y_test, y_pred, y_pred_proba)
        self.results['SVM'] = metrics
        
        # Save results
        self._save_model_results('SVM', metrics, output_dir)
        
        # Generate and save confusion matrix
        self._plot_confusion_matrix(self.y_test, y_pred, 'SVM', output_dir)
        
        return metrics

    def _save_model_results(self, model_name, metrics, output_dir):
        """
        Save model results to file
        """
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f'{model_name.lower()}_results.txt')
        
        with open(results_path, 'w') as f:
            f.write(f"{model_name} Model Results\n")
            f.write("=" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

    def _plot_confusion_matrix(self, y_true, y_pred, model_name, output_dir):
        """
        Plot and save confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_confusion_matrix.png'))
        plt.close()

    def plot_comparison(self, output_dir):
        """
        Create comparison plots for all models
        """
        if not self.results:
            print("No results to plot. Train models first.")
            return
            
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
        models = list(self.results.keys())
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        bar_width = 0.15
        index = np.arange(len(metrics))
        
        for i, model in enumerate(models):
            values = [self.results[model].get(metric, 0) for metric in metrics]
            plt.bar(index + i * bar_width, values, bar_width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(index + bar_width * (len(models)-1)/2, metrics)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(plot_path)
        plt.close()

def main():
    # Example usage
    data_path = '/root/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv'
    all_features = ['13', '14', '28', '29', 'manu_37',  '40', 'entropy13', 'entropy14', 'entropy28', 'entropy30', 'entropy14', 'entropy29']
    
    output_dir = '/root/Project/ClassifyProject/MOE/comparison/SVM/MCIvsAD'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing comparison framework...")
    # Initialize comparison framework
    comparison = ModelComparison(data_path, all_features)
    
    # Train and evaluate SVM
    print("\nStarting SVM training and evaluation...")
    svm_results = comparison.train_svm(output_dir)
    print("\nSVM Results:")
    for metric, value in svm_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    comparison.plot_comparison(output_dir)
    print(f"\nResults have been saved to: {output_dir}")

if __name__ == "__main__":
    main()