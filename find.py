import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.utils.class_weight import compute_class_weight
from data.dataset import AlzheimerDataset
from network.moe import MOE
from training.trainer import MoETrainer
from evaluation.evaluator import MoEEvaluator

def reset_random_seeds():
    """Reset all random seeds for reproducibility"""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(777)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(777)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(777)
    torch.use_deterministic_algorithms(True)

class FeatureSearcher:
    def __init__(self, data_path, labels, sex, device, num_epochs, batch_size):
        self.data_path = data_path
        self.labels = labels
        self.sex = sex
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # 탐색할 특징들
        self.all_features = ['manu_37', '29', 'entropy30', 'entropy29', '28', '14', '13', 'entropy14', 'entropy13', 'entropy28']
        
        self.best_combinations = []
        
    def evaluate_combination(self, expert1_features, expert2_features, model_dir, results_dir):
        """주어진 feature 조합을 평가"""
        reset_random_seeds()
        
        expert_features = {
            'expert_1': expert1_features,
            'expert_2': expert2_features
        }
        expert_dims = {name: len(features) for name, features in expert_features.items()}
        all_features = expert1_features + expert2_features
        
        # 데이터 로드
        dataset = AlzheimerDataset(
            data_path=self.data_path,
            interesting_features=all_features,
            labels=self.labels,
            sex=self.sex
        )
        
        data_splits = dataset.load_and_split_data()
        x_train, y_train, _ = data_splits['train']
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        dataloaders = dataset.data_loaders(batch_size=self.batch_size)
        
        # 모델 초기화
        model = MOE(expert_dims=expert_dims, out_dim=2).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-10, verbose=True
        )
        
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
        
        # 모델 학습
        trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=self.num_epochs,
            best_model_save_path=os.path.join(model_dir, f'moe_best_{len(self.best_combinations)}.pt'),
            final_model_save_path=os.path.join(model_dir, f'moe_final_{len(self.best_combinations)}.pt'),
            output_dir=results_dir
        )
        
        # 평가
        evaluator = MoEEvaluator(model=model, device=self.device, expert_features=expert_features)
        results = evaluator.evaluate(
            test_loader=dataloaders['test'],
            model_path=os.path.join(model_dir, f'moe_best_{len(self.best_combinations)}.pt'),
            output_dir=results_dir
        )
        
        return results, expert_features
    
    def search_combinations(self, min_features=1, max_features=None, top_k=5):
        """가능한 feature 조합들을 탐색"""
        if max_features is None:
            max_features = len(self.all_features) // 2  # 전체 특징의 절반까지만 한 expert에 할당
        
        model_dir = 'models/feature_search/'
        results_dir = 'results/feature_search/'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        for n_features in range(min_features, max_features + 1):
            print(f"\nTrying combinations with {n_features} features for expert 1...")
            
            # expert 1에 들어갈 특징 조합 생성
            for expert1_features in combinations(self.all_features, n_features):
                remaining_features = [f for f in self.all_features if f not in expert1_features]
                
                # expert 2에 들어갈 특징 조합 생성
                for n_features2 in range(min_features, len(remaining_features) + 1):
                    for expert2_features in combinations(remaining_features, n_features2):
                        print(f"\nTrying combination:")
                        print(f"Expert 1: {expert1_features}")
                        print(f"Expert 2: {expert2_features}")
                        
                        try:
                            results, expert_features = self.evaluate_combination(
                                list(expert1_features), 
                                list(expert2_features),
                                model_dir,
                                results_dir
                            )
                            
                            combination_result = {
                                'expert_features': expert_features,
                                'results': results,
                                'accuracy': results.get('accuracy', 0)
                            }
                            
                            self.best_combinations.append(combination_result)
                            self.best_combinations.sort(key=lambda x: x['accuracy'], reverse=True)
                            self.best_combinations = self.best_combinations[:top_k]
                            
                            print(f"Accuracy: {results.get('accuracy', 0):.4f}")
                            print(f"Current best accuracy: {self.best_combinations[0]['accuracy']:.4f}")
                            
                            # 현재까지의 결과 저장
                            self.save_results(os.path.join(results_dir, 'search_results.txt'))
                            
                        except Exception as e:
                            print(f"Error evaluating combination: {str(e)}")
                            continue
    
    def save_results(self, output_file):
        """탐색 결과를 파일로 저장"""
        with open(output_file, 'w') as f:
            f.write("Feature Combination Search Results\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(self.best_combinations, 1):
                f.write(f"Combination {i} (Accuracy: {result['accuracy']:.4f}):\n")
                f.write("-" * 30 + "\n")
                
                # Expert configuration
                for expert_name, features in result['expert_features'].items():
                    f.write(f"{expert_name}: {features}\n")
                
                # Results
                f.write("\nMetrics:\n")
                metric_order = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc']
                for metric_name in metric_order:
                    if metric_name in result['results']:
                        value = result['results'][metric_name]
                        f.write(f"{metric_name}: {value:.4f}\n")
                f.write("\n")

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reset_random_seeds()

    # Configuration
    labels = ['CN', 'AD']
    sex = 'A'
    num_epochs = 1
    batch_size = 32
    data_path = '/root/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv'

    # Initialize feature searcher
    searcher = FeatureSearcher(
        data_path=data_path,
        labels=labels,
        sex=sex,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    # Search for best combinations
    searcher.search_combinations(min_features=2, max_features=6, top_k=5)

if __name__ == "__main__":
    main()