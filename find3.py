import os
import torch
import torch.optim as optim
import numpy as np
import random
from itertools import permutations
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from data.dataset1 import AlzheimerDataset
from network.moe1 import ResidualMOE
from training.trainer1 import MoETrainer
from evaluation.evaluator1 import MoEEvaluator

def seed_worker(worker_id):
    """DataLoader worker 함수에 대한 시드 설정"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ExpertOrderOptimizer:
    def __init__(self, data_path, device, base_expert_features, base_dir, labels=['MCI', 'AD'], 
                 sex='A', num_epochs=10, batch_size=32):
        """
        Initialize the optimizer with fixed expert groupings
        
        Args:
            data_path: Path to the dataset
            device: Computation device (cuda/cpu)
            base_expert_features: Dictionary with initial expert feature groupings
            base_dir: Base directory for all outputs
            labels: List of class labels
            sex: Sex filter for dataset
            num_epochs: Number of epochs for each training run
            batch_size: Batch size for training
        """
        self.data_path = data_path
        self.device = device
        self.base_expert_features = base_expert_features
        self.base_dir = base_dir
        self.labels = labels
        self.sex = sex
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Setup directories
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_ordering(self, expert_features):
        """Evaluate a specific ordering of features within experts"""
        try:
            # Get all features in current order
            all_features = []
            for features in expert_features.values():
                all_features.extend(features)

            # Create dataset
            dataset = AlzheimerDataset(
                data_path=self.data_path,
                interesting_features=all_features,
                labels=self.labels,
                sex=self.sex
            )

            data_splits = dataset.load_and_split_data()
            x_train = data_splits['train'][0]
            y_train = data_splits['train'][1]
            
            print(f"x_train shape: {x_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            
            # Compute class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            
            # expert_features를 data_loaders에 전달
            dataloaders = dataset.data_loaders(
                batch_size=self.batch_size,
                expert_features=expert_features
            )

            # Setup model
            torch.manual_seed(777)
            expert_dims = {name: len(features) for name, features in expert_features.items()}
            model = ResidualMOE(expert_dims=expert_dims, out_dim=2).to(self.device)

            if hasattr(model, 'init_weights'):
                model.init_weights()
            
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-10
            )

            # Initialize trainer
            trainer = MoETrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                expert_features=expert_features,
                alpha=0.05,
                beta=0.05,
                class_weights=class_weights
            )

            # Train model
            best_val_loss, best_val_acc = trainer.train(
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                num_epochs=self.num_epochs,
                best_model_save_path=os.path.join(self.model_dir, 'moe_best.pt'),
                final_model_save_path=os.path.join(self.model_dir, 'moe_final.pt'),
                output_dir=self.results_dir
            )

            # Evaluate model
            evaluator = MoEEvaluator(model=model, device=self.device, expert_features=expert_features)
            results = evaluator.evaluate(
                test_loader=dataloaders['test'],
                model_path=os.path.join(self.model_dir, 'moe_best.pt'),
                output_dir=self.results_dir
            )

            return results

        except Exception as e:
            print(f"Error in evaluate_ordering: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Stack trace:\n{traceback.format_exc()}")
            
            # 추가 디버깅 정보
            if 'data_splits' in locals():
                print("\nData splits info:")
                for split_name, split_data in data_splits.items():
                    print(f"{split_name}: type={type(split_data)}, ", end="")
                    if isinstance(split_data, (list, tuple)):
                        print(f"length={len(split_data)}")
                    else:
                        print("not a sequence")
            return None

    def optimize_expert_order(self):
            """Find optimal feature ordering within each expert"""
            best_accuracy = -1
            best_ordering = None
            best_results = None
            
            # Store all results for analysis
            all_results = []

            # Generate permutations for each expert
            expert1_perms = list(permutations(self.base_expert_features['expert_1']))
            expert2_perms = list(permutations(self.base_expert_features['expert_2']))

            total_combinations = len(expert1_perms) * len(expert2_perms)
            print(f"Total combinations to try: {total_combinations}")
            print("\nCurrent Best Configuration:")
            print("-" * 50)
            print(f"{'Combination':<15} {'Accuracy':<10} {'Expert 1 Order':<50} {'Expert 2 Order'}")
            print("-" * 120)

            try:
                for i, perm1 in enumerate(expert1_perms, 1):
                    for j, perm2 in enumerate(expert2_perms, 1):
                        current_count = (i-1) * len(expert2_perms) + j
                        print(f"\nTrying combination {current_count}/{total_combinations}")
                        print(f"Expert 1 order: {perm1}")
                        print(f"Expert 2 order: {perm2}")

                        # Create current feature configuration
                        current_features = {
                            'expert_1': list(perm1),
                            'expert_2': list(perm2)
                        }

                        # Evaluate current ordering
                        results = self.evaluate_ordering(current_features)
                        
                        if results is not None:
                            current_accuracy = results['accuracy']
                            
                            # Store results
                            result_entry = {
                                'combination_number': current_count,
                                'expert_1_order': list(perm1),
                                'expert_2_order': list(perm2),
                                'metrics': results
                            }
                            all_results.append(result_entry)

                            # Update best if necessary
                            if current_accuracy > best_accuracy:
                                best_accuracy = current_accuracy
                                best_ordering = current_features
                                best_results = results
                                
                                # Clear previous line and update best configuration display
                                print("\033[F" * 2)  # Move cursor up 2 lines
                                print("-" * 120)
                                print(f"{current_count:<15} {current_accuracy:<10.4f} {str(list(perm1)):<50} {str(list(perm2))}")
                                print(f"\nNew best ordering found! Accuracy: {current_accuracy:.4f}")
                                
                                # Display other metrics
                                print("\nCurrent Best Metrics:")
                                metric_order = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc']
                                for metric_name in metric_order:
                                    if metric_name in results:
                                        print(f"{metric_name}: {results[metric_name]:.4f}")

                            # Save progress
                            self.save_progress(all_results, best_ordering, best_results)

            except KeyboardInterrupt:
                print("\nOptimization interrupted by user!")
            finally:
                # Save final results
                self.save_final_results(best_ordering, best_results, all_results)

            return best_ordering, best_results

    def save_progress(self, all_results, best_ordering, best_results):
        """Save current progress to file"""
        progress = {
            'all_results': all_results,
            'best_ordering': best_ordering,
            'best_results': best_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open(os.path.join(self.results_dir, 'optimization_progress.json'), 'w') as f:
            json.dump(progress, f, indent=4)

    def save_final_results(self, best_ordering, best_results, all_results):
        """Save final optimization results"""
        with open(os.path.join(self.results_dir, 'final_results.txt'), 'w') as f:
            f.write("Expert Feature Order Optimization Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Best Feature Ordering:\n")
            for expert_name, features in best_ordering.items():
                f.write(f"{expert_name}: {features}\n")
            
            f.write("\nBest Metrics:\n")
            for metric_name, value in best_results.items():
                f.write(f"{metric_name}: {value:.4f}\n")

            # Save detailed results for all combinations
            import pandas as pd
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(self.results_dir, 'all_combinations_results.csv'), 
                            index=False)

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set base directory
    base_dir = '/root/Project/ClassifyProject/MOE/results/search_residual/MCIvsAD'
    
    # Deterministic settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(777)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    
    torch.use_deterministic_algorithms(True)

    # Initial expert feature groupings
    base_expert_features = {
        'expert_1': ['manu_37', '40', '13', '29', '14', '28', 'Sum'],
        'expert_2': ['entropy30', 'entropy29', 'entropy28', 'entropy13', 'entropy14']
    }

    # Initialize optimizer
    optimizer = ExpertOrderOptimizer(
        data_path='/root/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv',
        device=device,
        base_expert_features=base_expert_features,
        base_dir=base_dir,
        num_epochs=1,
        batch_size=32
    )

    # Run optimization
    best_ordering, best_results = optimizer.optimize_expert_order()

    print("\nOptimization Complete!")
    print("\nBest Feature Ordering:")
    for expert_name, features in best_ordering.items():
        print(f"{expert_name}: {features}")
    
    print("\nBest Metrics:")
    for metric_name, value in best_results.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()