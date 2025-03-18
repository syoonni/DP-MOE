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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from data.dataset import AlzheimerDataset

class DecisionTreeTrainer:
    """Decision Tree implementation with detailed training and visualization"""
    
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
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Validation set size: {len(self.X_val)}")
        print(f"Test set size: {len(self.X_test)}")

    def _optimize_hyperparameters(self):
        """Optimize hyperparameters using GridSearchCV"""
        print("\nOptimizing hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        # Initialize base model
        base_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print("\nBest parameters found:", grid_search.best_params_)
        return grid_search.best_params_
    
    def train(self, output_dir):
        """Train the decision tree model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimize hyperparameters
        best_params = self._optimize_hyperparameters()
        
        # Train model with best parameters
        print("\nTraining final model with best parameters...")
        self.model = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            **best_params
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(self.X_val)
        val_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        val_metrics = {
            'accuracy': accuracy_score(self.y_val, val_pred),
            'precision': precision_score(self.y_val, val_pred),
            'recall': recall_score(self.y_val, val_pred),
            'f1': f1_score(self.y_val, val_pred),
            'specificity': recall_score(self.y_val, val_pred, pos_label=0),
            'auroc': roc_auc_score(self.y_val, val_pred_proba)
        }
        
        print("\nValidation Set Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Evaluate on test set
        test_pred = self.model.predict(self.X_test)
        test_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(self.y_test, test_pred_proba)

        # Save ROC data to CSV
        roc_data = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        })
        roc_data.to_csv(os.path.join(output_dir, 'roc_data.csv'), index=False)
        
        test_metrics = {
            'accuracy': accuracy_score(self.y_test, test_pred),
            'precision': precision_score(self.y_test, test_pred),
            'recall': recall_score(self.y_test, test_pred),
            'f1': f1_score(self.y_test, test_pred),
            'auroc': roc_auc_score(self.y_test, test_pred_proba)
        }
        
        # Save results and visualizations
        self._save_results(val_metrics, test_metrics, best_params, output_dir)
        self._plot_decision_tree(output_dir)
        self._plot_feature_importance(output_dir)
        self._plot_confusion_matrix(test_pred, output_dir)

        # Plot and save ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {test_metrics["auroc"]:.2f})')
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
    
    def _save_results(self, val_metrics, test_metrics, best_params, output_dir):
        """Save all results to file"""
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write("Decision Tree Results\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            
            f.write("\nValidation Set Metrics:\n")
            for metric, value in val_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nTest Set Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            # Save tree structure as text
            f.write("\nTree Structure:\n")
            f.write(export_text(self.model, feature_names=self.all_features))
    
    def _plot_decision_tree(self, output_dir):
        """Plot and save the decision tree visualization"""
        plt.figure(figsize=(20,10))
        plot_tree(self.model, 
                 feature_names=self.all_features,
                 class_names=self.labels,
                 filled=True,
                 rounded=True)
        plt.savefig(os.path.join(output_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, output_dir):
        """Plot and save feature importance"""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.all_features,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance)), feature_importance['importance'])
        plt.yticks(range(len(importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance in Decision Tree')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), 
                                index=False)
    
    def _plot_confusion_matrix(self, y_pred, output_dir):
        """Plot and save confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

def main():
    # Configuration
    data_path = '/root/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv'
    all_features = ['13', '14', '28', '29', 'manu_37', 'entropy13', 'entropy14', 'entropy28', 'entropy29', '40', 'entropy30']
    
    output_dir = '/root/Project/ClassifyProject/MOE/comparison/RF'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    print("Initializing Decision Tree trainer...")
    trainer = DecisionTreeTrainer(data_path, all_features)
    
    # Train and evaluate
    print("\nStarting training and evaluation...")
    results = trainer.train(output_dir)
    
    print("\nFinal Test Set Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nResults and visualizations have been saved to: {output_dir}")

if __name__ == "__main__":
    main()