import os
import sys

# Add parent directory to python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap

from data.dataset import AlzheimerDataset

class XGBoostTrainer:
    """XGBoost implementation with detailed training and visualization"""
    
    def __init__(self, data_path, all_features, labels=['MCI', 'AD'], sex='A'):
        self.data_path = data_path
        self.all_features = all_features
        self.labels = labels
        self.sex = sex
        
        print("Loading and preprocessing data...")
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """Load and preprocess the data"""
        # Load dataset
        self.dataset = AlzheimerDataset(
            data_path=self.data_path,
            interesting_features=self.all_features,
            labels=self.labels,
            sex=self.sex
        )
        
        # Get data splits
        data_splits = self.dataset.load_and_split_data()
        
        # Extract data
        self.X_train = data_splits['train'][0]
        self.y_train = data_splits['train'][1]
        self.X_val = data_splits['val'][0]
        self.y_val = data_splits['val'][1]
        self.X_test = data_splits['test'][0]
        self.y_test = data_splits['test'][1]
        
        # Convert to DMatrix format
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train, 
                                 feature_names=self.all_features)
        self.dval = xgb.DMatrix(self.X_val, label=self.y_val, 
                               feature_names=self.all_features)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test, 
                                feature_names=self.all_features)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Validation set size: {len(self.X_val)}")
        print(f"Test set size: {len(self.X_test)}")

    def train(self, output_dir):
            """Train the XGBoost model"""
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate class weights
            pos_scale = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
            
            # Set parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['error', 'logloss'],  # Removed AUC temporarily
                'max_depth': 4,
                'min_child_weight': 1,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': pos_scale,
                'tree_method': 'hist',
                'random_state': 42
            }
            
            print("\nStarting XGBoost training...")
            print(f"Class balance ratio (Negative/Positive): {pos_scale:.2f}")
            
            # Create evaluation list for tracking metrics
            evallist = [(self.dtrain, 'train'), (self.dval, 'eval')]
            
            # Dictionary to store evaluation results
            evals_result = {}
            
            # Train model with early stopping
            self.model = xgb.train(
                params,
                self.dtrain,
                num_boost_round=1000,
                evals=evallist,
                early_stopping_rounds=50,
                verbose_eval=10,
                evals_result=evals_result  # Store evaluation results
            )
            
            # Store evaluation results for later use
            self.evals_result = evals_result
            
            print(f"\nBest iteration: {self.model.best_iteration}")
            
            # Evaluate on validation set
            val_pred = self.model.predict(self.dval)
            val_pred_binary = (val_pred > 0.5).astype(int)
            
            val_metrics = {
                'accuracy': accuracy_score(self.y_val, val_pred_binary),
                'precision': precision_score(self.y_val, val_pred_binary),
                'recall': recall_score(self.y_val, val_pred_binary),
                'f1': f1_score(self.y_val, val_pred_binary),
                'specificity': recall_score(self.y_val, val_pred_binary, pos_label=0)
            }
            
            # Only calculate AUROC if both classes are present
            if len(np.unique(self.y_val)) > 1:
                val_metrics['auroc'] = roc_auc_score(self.y_val, val_pred)
            else:
                val_metrics['auroc'] = None
            
            print("\nValidation Set Metrics:")
            for metric, value in val_metrics.items():
                if value is not None:
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: N/A")
            
            # Evaluate on test set
            test_pred = self.model.predict(self.dtest)
            test_pred_binary = (test_pred > 0.5).astype(int)

            # Calculate ROC curve data
            fpr, tpr, thresholds = roc_curve(self.y_test, test_pred_binary)

            # Save ROC data to CSV
            roc_data = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
            roc_data.to_csv(os.path.join(output_dir, 'roc_data.csv'), index=False)
            
            test_metrics = {
                'accuracy': accuracy_score(self.y_test, test_pred_binary),
                'precision': precision_score(self.y_test, test_pred_binary),
                'recall': recall_score(self.y_test, test_pred_binary),
                'f1': f1_score(self.y_test, test_pred_binary)
            }
            
            # Only calculate AUROC if both classes are present
            if len(np.unique(self.y_test)) > 1:
                test_metrics['auroc'] = roc_auc_score(self.y_test, test_pred)
            else:
                test_metrics['auroc'] = None
            
            # Save results and visualizations
            self._save_results(val_metrics, test_metrics, params, output_dir)
            self._plot_feature_importance(output_dir)
            self._plot_learning_curves(output_dir)
            self._plot_confusion_matrix(test_pred_binary, output_dir)
            self._save_shap_analysis(output_dir)

            # Plot ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc_score(self.y_test, test_pred_binary):.2f})')
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
            
            return test_metrics
    
    def _save_results(self, val_metrics, test_metrics, params, output_dir):
        """Save all results to file"""
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write("XGBoost Results\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Model Parameters:\n")
            for param, value in params.items():
                f.write(f"{param}: {value}\n")
            
            f.write(f"\nBest Iteration: {self.model.best_iteration}\n")
            
            f.write("\nValidation Set Metrics:\n")
            for metric, value in val_metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
                else:
                    f.write(f"{metric}: N/A\n")
            
            f.write("\nTest Set Metrics:\n")
            for metric, value in test_metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
                else:
                    f.write(f"{metric}: N/A\n")
    
    def _plot_feature_importance(self, output_dir):
        """Plot and save feature importance"""
        importance_type = 'gain'  # Can be 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        
        # Get feature importance
        importance = self.model.get_score(importance_type=importance_type)
        importance = pd.DataFrame(
            [(k, v) for k, v in importance.items()],
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=True)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance)), importance['importance'])
        plt.yticks(range(len(importance)), importance['feature'])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance to CSV
        importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), 
                         index=False)
    
    def _plot_learning_curves(self, output_dir):
            """Plot and save learning curves"""
            if not hasattr(self, 'evals_result'):
                print("No evaluation results found. Skipping learning curves plot.")
                return
                
            plt.figure(figsize=(12, 4))
            
            # Plot error
            plt.subplot(1, 2, 1)
            for dataset in ['train', 'eval']:
                plt.plot(
                    self.evals_result[dataset]['error'],
                    label=f'{dataset}'
                )
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title('Classification Error')
            plt.legend()
            
            # Plot log loss
            plt.subplot(1, 2, 2)
            for dataset in ['train', 'eval']:
                plt.plot(
                    self.evals_result[dataset]['logloss'],
                    label=f'{dataset}'
                )
            plt.xlabel('Iteration')
            plt.ylabel('Log Loss')
            plt.title('Log Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
            plt.close()

    def _plot_confusion_matrix(self, y_pred, output_dir):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    
    def _save_shap_analysis(self, output_dir):
        """Perform and save SHAP analysis"""
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.all_features,
                            show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
            plt.close()
            
            # Feature interaction plot for top features
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(0, shap_values, self.X_test, 
                               feature_names=self.all_features,
                               show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_interaction.png'))
            plt.close()
            
        except ImportError:
            print("SHAP package not found. Skipping SHAP analysis.")

def main():
    # Configuration
    data_path = '/root/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv'
    all_features = ['13', '14', '28', '29', 'manu_37',  '40', 'entropy13', 'entropy14', 'entropy28', 'entropy30', 'entropy29']
    
    output_dir = '/root/Project/ClassifyProject/MOE/comparison/SVM/MCIvsAD'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    print("Initializing XGBoost trainer...")
    trainer = XGBoostTrainer(data_path, all_features)
    
    # Train and evaluate
    print("\nStarting training and evaluation...")
    results = trainer.train(output_dir)
    
    print("\nFinal Test Set Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nResults and visualizations have been saved to: {output_dir}")

if __name__ == "__main__":
    main()