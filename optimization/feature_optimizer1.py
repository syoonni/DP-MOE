import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class FeatureOptimizer:
    def __init__(self, dataloaders, all_features, device, min_experts=2, max_experts=3):
        self.dataloaders = dataloaders
        self.all_features = all_features
        self.device = device
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.train_data = None
        self.train_labels = None
        self._extract_training_data()

    def _extract_training_data(self):
        """Extract training data from dataloader for analysis"""
        data_iter = iter(self.dataloaders['train'])
        batch = next(data_iter)
        self.train_data = batch[0].numpy()
        self.train_labels = batch[1].numpy()

    def compute_feature_similarity_matrix(self):
        """Compute feature similarity matrix using correlation"""
        # Correlation-based similarity
        corr_matrix = np.corrcoef(self.train_data.T)
        # Convert to absolute values and scale to [0, 1]
        similarity_matrix = (np.abs(corr_matrix) + 1) / 2
        return similarity_matrix

    def cluster_features(self, similarity_matrix, n_experts):
        """Cluster features using hierarchical clustering"""
        # Convert similarity to distance (ensuring non-negative values)
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.abs(distance_matrix)  # Ensure non-negative
        np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0
        
        # Normalize distance matrix to [0, 1]
        if distance_matrix.max() > 0:
            distance_matrix = distance_matrix / distance_matrix.max()
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_experts,
            metric='precomputed',
            linkage='complete'
        )
        
        clusters = clustering.fit_predict(distance_matrix)
        
        # Skip silhouette score calculation if we have only 2 features or less
        if len(self.all_features) <= 2:
            silhouette_avg = 0
        else:
            try:
                silhouette_avg = silhouette_score(
                    distance_matrix,
                    clusters,
                    metric='precomputed'
                )
            except:
                print("Warning: Silhouette score calculation failed, using default value")
                silhouette_avg = 0
        
        # Group features by cluster
        feature_groups = {}
        for i in range(n_experts):
            feature_indices = np.where(clusters == i)[0]
            if len(feature_indices) == 0:  # Handle empty clusters
                print(f"Warning: Empty cluster {i} detected")
                continue
                
            feature_groups[f'expert_{i+1}'] = [
                self.all_features[j] for j in feature_indices
            ]
        
        # Handle case where some clusters are empty
        if len(feature_groups) < n_experts:
            print("Warning: Some clusters were empty, using simpler grouping")
            features_per_expert = len(self.all_features) // n_experts
            feature_groups = {}
            for i in range(n_experts):
                start_idx = i * features_per_expert
                end_idx = start_idx + features_per_expert if i < n_experts - 1 else None
                feature_groups[f'expert_{i+1}'] = self.all_features[start_idx:end_idx]
        
        return feature_groups, silhouette_avg

    def evaluate_expert_config(self, feature_groups):
        """Evaluate a feature grouping configuration"""
        if not feature_groups:  # Handle empty configuration
            return 0.0
            
        # Prepare data for each expert
        expert_data = []
        expert_dims = []
        
        for expert_features in feature_groups.values():
            if not expert_features:  # Skip empty feature groups
                continue
            indices = [self.all_features.index(f) for f in expert_features]
            expert_data.append(torch.FloatTensor(self.train_data[:, indices]))
            expert_dims.append(len(indices))

        if not expert_data:  # Handle case where no valid expert data
            return 0.0

        # Create simple evaluation model
        class EvalModel(nn.Module):
            def __init__(self, expert_dims):
                super().__init__()
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)
                    ) for dim in expert_dims
                ])
                
            def forward(self, x_list):
                expert_outputs = [expert(x) for expert, x in zip(self.experts, x_list)]
                return torch.stack(expert_outputs).mean(0)

        try:
            # Create and train evaluation model
            model = EvalModel(expert_dims).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Quick training loop
            for epoch in range(5):
                model.train()
                expert_inputs = [data.to(self.device) for data in expert_data]
                labels = torch.LongTensor(self.train_labels).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(expert_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                expert_inputs = [data.to(self.device) for data in expert_data]
                outputs = model(expert_inputs)
                preds = outputs.argmax(1)
                accuracy = (preds == labels).float().mean().item()

            # Calculate balance score
            sizes = [len(features) for features in feature_groups.values()]
            if len(sizes) > 1:
                balance_score = 1 - np.std(sizes) / np.mean(sizes)
            else:
                balance_score = 0

            return 0.7 * accuracy + 0.3 * balance_score
            
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {str(e)}")
            return 0.0

    def optimize(self):
        """Find optimal feature grouping"""
        print("Computing feature similarity matrix...")
        similarity_matrix = self.compute_feature_similarity_matrix()

        best_score = -float('inf')
        best_config = None
        best_n_experts = None

        # Try different numbers of experts
        for n_experts in range(self.min_experts, self.max_experts + 1):
            print(f"\nTrying {n_experts} experts...")
            
            try:
                # Cluster features
                feature_groups, silhouette_avg = self.cluster_features(
                    similarity_matrix, 
                    n_experts
                )
                
                # Evaluate configuration
                config_score = self.evaluate_expert_config(feature_groups)
                total_score = 0.7 * config_score + 0.3 * silhouette_avg
                
                print(f"Configuration score: {total_score:.4f}")
                print("Feature groups:")
                for name, features in feature_groups.items():
                    print(f"{name}: {features}")

                if total_score > best_score:
                    best_score = total_score
                    best_config = feature_groups
                    best_n_experts = n_experts
                    
            except Exception as e:
                print(f"Warning: Failed to evaluate {n_experts} experts configuration: {str(e)}")
                continue

        if best_config is None:
            print("\nWarning: Optimization failed, using default configuration")
            mid = len(self.all_features) // 2
            best_config = {
                'expert_1': self.all_features[:mid],
                'expert_2': self.all_features[mid:]
            }
            best_n_experts = 2

        return best_config, best_n_experts