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

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reset_random_seeds()

    # Configuration
    labels = ['CN', 'AD']
    sex = 'A'
    num_epochs = 100
    batch_size = 32

    # Define expert feature groups
    expert_features = {
        'expert_1': ['Amygdala(rh)-vol', 'Amygdala(lh)-vol', 'Entorhinal-vol', 'Hippocampus(rh)-vol', 'Hippocampus(lh)-vol'],
        'expert_2': ['Amygdala(rh)-entropy', 'Amygdala(lh)-entropy', 'Entorhinal-entropy', 'Hippocampus(rh)-entropy', 'Hippocampus(lh)-entropy']  
    }

    # Calculate dimensions for each expert
    expert_dims = {name: len(features) for name, features in expert_features.items()}

    # Directories
    model_dir = 'models/all_cnvsad/'
    results_dir = 'results/all_cnvsad/'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data paths and features
    data_path = 'data.csv'

    # Combine all features for dataset loading
    all_features = []
    for features in expert_features.values():
        all_features.extend(features)

    # Load and preprocess data
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

    # Initialize MoE model
    model = MOE(expert_dims=expert_dims, out_dim=2).to(device)
    
    # Setup optimizer and scheduler
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
        best_model_save_path=os.path.join(model_dir, 'moe_best.pt'),
        final_model_save_path=os.path.join(model_dir, 'moe_final.pt'),
        output_dir=results_dir
    )

    # Evaluate the model
    evaluator = MoEEvaluator(model=model, device=device, expert_features=expert_features)
    results = evaluator.evaluate(
        test_loader=dataloaders['test'],
        model_path=os.path.join(model_dir, 'moe_best.pt'),
        output_dir=results_dir
    )

    # Print final results
    print("\nFinal Evaluation Results:")
    with open(os.path.join(results_dir, 'final_results.txt'), 'w') as f:
        f.write("MoE Model Final Results:\n")
        f.write("-" * 30 + "\n")
        f.write("Performance Metrics:\n")

        metric_order = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc']
        for metric_name in metric_order:
            if metric_name in results:
                value = results[metric_name]
                print(f"{metric_name}: {value:.4f}")
                f.write(f"{metric_name}: {value:.4f}\n")

        # Add expert configuration information
        f.write("\nExpert Configuration:\n")
        f.write("-" * 30 + "\n")
        for expert_name, features in expert_features.items():
            f.write(f"{expert_name}: {features}\n")

if __name__ == "__main__":
    main()
