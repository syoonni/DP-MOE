import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data.dataset import AlzheimerDataset
from network.moe import MOE
from training.trainer import MoETrainer
from evaluation.evaluator import MoEEvaluator
from optimization.feature_optimizer1 import FeatureOptimizer

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # Configuration
    labels = ['AD', 'CN']
    sex = 'A'
    num_epochs = 150
    batch_size = 32

    # Data paths and all available features
    data_path = '/home/syoon/Project/ClassifyProject/Ensemble/AlzheimerDataset/fast_data + manual.csv'
    all_features = ['manu_37', '13', '14', '29', '28', '40', 'Sum', 'entropy30', 'entropy29', 'entropy28', 'entropy13', 'entropy14']

    # Load dataset first
    dataset = AlzheimerDataset(
        data_path=data_path,
        interesting_features=all_features,
        labels=labels,
        sex=sex
    )

    data_splits = dataset.load_and_split_data()
    x_train, y_train, _ = data_splits['train']
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Classes:", classes)
    print("Class Weights (before converting to tensor):", compute_class_weight('balanced', classes=classes, y=y_train))
    print("Class Weights (as tensor):", class_weights)
    
    dataloaders = dataset.data_loaders(batch_size=batch_size)

    # Initialize and run Feature Optimizer
    print("\nOptimizing feature combinations...")
    optimizer = FeatureOptimizer(
        dataloaders=dataloaders,
        all_features=all_features,
        device=device,
        min_experts=2,
        max_experts=4
    )
    
    expert_features, n_experts = optimizer.optimize()
    
    print("\nOptimal Expert Configuration found:")
    for name, features in expert_features.items():
        print(f"{name}: {features}")

    # Calculate dimensions for each expert
    expert_dims = {name: len(features) for name, features in expert_features.items()}

    # Directories
    model_dir = 'models/vol+entropy/'
    results_dir = 'results/vol+entropy/'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save feature optimization results
    with open(os.path.join(results_dir, 'feature_optimization.txt'), 'w') as f:
        f.write("Optimized Feature Configuration:\n")
        f.write("-" * 30 + "\n")
        for name, features in expert_features.items():
            f.write(f"{name}: {features}\n")

    # Initialize MoE model
    model = MOE(expert_dims=expert_dims, out_dim=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=1e-10,
        verbose=True
    )

    # Initialize trainer
    trainer = MoETrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        expert_features=expert_features,
        alpha=0.05,
        beta=0.05,
        class_weights=class_weights
    )

    # Train the model
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=num_epochs,
        best_model_save_path=os.path.join(model_dir, 'best_model.pt'),
        final_model_save_path=os.path.join(model_dir, 'final_model.pt'),
        output_dir=results_dir
    )

    # Evaluate the model
    evaluator = MoEEvaluator(model=model, device=device, expert_features=expert_features)

    results = evaluator.evaluate(
        test_loader=dataloaders['test'],
        model_path=os.path.join(model_dir, 'best_model.pt'),
        output_dir=results_dir
    )

    print("\nFinal Evaluation Results:")
    with open(os.path.join(results_dir, 'final_results.txt'), 'w') as f:
        f.write("MoE Model Final Results:\n")
        f.write("-" * 30 + "\n")
        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")
            f.write(f"{metric_name}: {value:.4f}\n")

        # Add expert configuration information
        f.write("\nExpert Configuration:\n")
        f.write("-" * 30 + "\n")
        for expert_name, features in expert_features.items():
            f.write(f"{expert_name}: {features}\n")

if __name__ == "__main__":
    main()