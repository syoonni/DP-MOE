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
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from data.dataset import AlzheimerDataset

class RandomForestTrainer:
    """Random Forest implementation with detailed training and visualization"""
    
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
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        # Initialize base model
        base_model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            oob_score=True,  # Enable out-of-bag scoring
            n_jobs=-1  # Use all CPU cores
        )
        
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
        """Train the Random Forest model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimize hyperparameters
        best_params = self._optimize_hyperparameters()
        
        # Train model with best parameters
        print("\nTraining final model with best parameters...")
        self.model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            oob_score=True,
            n_jobs=-1,
            **best_params
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Print OOB score
        print(f"\nOut-of-Bag Score: {self.model.oob_score_:.4f}")
        
        # Evaluate on validation set
        val_pred = self.model.predict(self.X_val)
        val_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        val_metrics = {
            'accuracy': accuracy_score(self.y_val, val_pred),
            'precision': precision_score(self.y_val, val_pred),
            'recall': recall_score(self.y_val, val_pred),
            'f1': f1_score(self.y_val, val_pred),
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
            'specificity': recall_score(self.y_test, test_pred, pos_label=0),
            'auroc': roc_auc_score(self.y_test, test_pred_proba)
        }
        
        # Save results and visualizations
        self._save_results(val_metrics, test_metrics, best_params, output_dir)
        self._plot_feature_importance(output_dir)
        self._plot_confusion_matrix(test_pred, output_dir)
        self._analyze_feature_interactions(output_dir)

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
            f.write("Random Forest Results\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            
            f.write(f"\nOut-of-Bag Score: {self.model.oob_score_:.4f}\n")
            
            f.write("\nValidation Set Metrics:\n")
            for metric, value in val_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nTest Set Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    
    def _plot_feature_importance(self, output_dir):
        """Plot and save feature importance"""
        # Get feature importance from all trees
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': self.all_features,
            'importance': importances,
            'std': std
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        # Plot feature importance with error bars
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), feature_importance['importance'],
                xerr=feature_importance['std'], align='center')
        plt.yticks(range(len(importances)), feature_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance in Random Forest')
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
    
    def _analyze_feature_interactions(self, output_dir):
        """Analyze and visualize feature interactions"""
        # Get the top two most important features
        importances = self.model.feature_importances_
        top_features_idx = np.argsort(importances)[-2:]
        top_features = [self.all_features[i] for i in top_features_idx]
        
        # Plot partial dependence for top two features
        plt.figure(figsize=(10, 5))
        
        # Plot histograms of feature values separated by class
        for i, feature in enumerate(top_features):
            plt.subplot(1, 2, i+1)
            feature_idx = self.all_features.index(feature)
            
            for label in [0, 1]:
                mask = self.y_train == label
                plt.hist(self.X_train[mask, feature_idx], 
                        bins=20, 
                        alpha=0.5,
                        label=f'Class {label}')
            
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_interactions.png'))
        plt.close()

def main():
    # Configuration
    data_path = '/root/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv'
    all_features = ['13', '14', '28', '29', 'manu_37',  '40', 'entropy13', 'entropy14', 'entropy28', 'entropy30', 'entropy14', 'entropy29']
    
    output_dir = '/root/Project/ClassifyProject/MOE/comparison/RF/MCIvsAD'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    print("Initializing Random Forest trainer...")
    trainer = RandomForestTrainer(data_path, all_features)
    
    # Train and evaluate
    print("\nStarting training and evaluation...")
    results = trainer.train(output_dir)
    
    print("\nFinal Test Set Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nResults and visualizations have been saved to: {output_dir}")

if __name__ == "__main__":
    main()