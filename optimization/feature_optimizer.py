import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from network.moe import Expert
import itertools

class FeatureOptimizer:
    """Optimize feature grouping for MoE experts using correlation and performance metrics"""
    
    def __init__(self, dataloaders, all_features, device, n_splits=5):
        """
        Initialize the optimizer
        
        Args:
            dataloaders: Dictionary containing train/val/test dataloaders
            all_features: List of all available feature names
            device: torch device (cuda/cpu)
            n_splits: Number of folds for cross validation
        """
        self.dataloaders = dataloaders
        self.all_features = all_features
        self.device = device
        self.n_splits = n_splits
        
    def compute_feature_correlation(self):
        """Compute correlation matrix between features"""
        # Extract features from training data
        train_data = next(iter(self.dataloaders['train']))[0].numpy()
        
        # Create correlation matrix
        corr_matrix = np.corrcoef(train_data.T)
        return pd.DataFrame(corr_matrix, columns=self.all_features, index=self.all_features)
    
    def cluster_features(self, corr_matrix, threshold=0.5):
        """
        Group features based on correlation threshold
        Returns list of possible feature groupings
        """
        remaining_features = set(self.all_features)
        feature_groups = []
        
        while remaining_features:
            current_feature = remaining_features.pop()
            current_group = {current_feature}
            
            for feature in list(remaining_features):
                correlation = abs(corr_matrix.loc[current_feature, feature])
                if correlation > threshold:
                    current_group.add(feature)
                    remaining_features.remove(feature)
                    
            feature_groups.append(list(current_group))
            
        return feature_groups
        
    def evaluate_expert(self, features, X_train, y_train, X_val, y_val):
        """
        Evaluate performance of a single expert with given features
        
        Returns:
            float: validation accuracy
        """
        input_dim = len(features)
        expert = Expert(input_dim, 2).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(expert.parameters(), lr=0.001)
        
        # Convert data to torch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Train for a few epochs
        for epoch in range(10):
            expert.train()
            optimizer.zero_grad()
            outputs, _ = expert(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
        # Evaluate
        expert.eval()
        with torch.no_grad():
            val_outputs, _ = expert(X_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1)
            accuracy = accuracy_score(y_val_tensor.cpu(), val_preds.cpu())
            
        return accuracy
        
    def generate_expert_configs(self, min_experts=2, max_experts=3):
        """
        Generate optimal expert configurations
        
        Returns:
            dict: Expert configurations with feature assignments
            int: Number of experts
        """
        # Default configuration in case optimization fails
        default_expert_features = {
            'expert_1': self.all_features[:len(self.all_features)//2],
            'expert_2': self.all_features[len(self.all_features)//2:]
        }
        
        try:
            # Compute correlation matrix
            corr_matrix = self.compute_feature_correlation()
            
            # Get initial feature groups based on correlation
            feature_groups = self.cluster_features(corr_matrix)
            
            if not feature_groups:
                print("Warning: No feature groups found. Using default configuration.")
                return default_expert_features, 2
            
            # Generate possible combinations of feature groups
            best_config = None
            best_score = 0
            best_n_experts = 0
            
            # Try different numbers of experts
            for n_experts in range(min_experts, max_experts + 1):
                print(f"\nTrying {n_experts} experts...")
                
                # Generate possible feature combinations
                possible_combinations = []
                for group in feature_groups:
                    if len(group) > 1:
                        # Consider both keeping group together and splitting
                        possible_combinations.append([group])
                        possible_combinations.append([[f] for f in group])
                    else:
                        possible_combinations.append([group])
                
                # Try different combinations of feature groups
                for combination in itertools.product(*possible_combinations):
                    # Flatten and ensure no feature is used twice
                    flat_features = [f for group in combination for f in group]
                    if len(flat_features) != len(set(flat_features)):
                        continue
                        
                    if len(flat_features) != len(self.all_features):
                        continue
                    
                    # Group features into n_experts groups
                    feature_assignment = self._assign_features_to_experts(flat_features, n_experts)
                    
                    # Evaluate this configuration
                    score = self._evaluate_config(feature_assignment)
                    
                    if score > best_score:
                        best_score = score
                        best_config = feature_assignment
                        best_n_experts = n_experts
                        
                    print(f"Configuration score: {score:.4f}")
        
        except Exception as e:
            print(f"Warning: Optimization failed with error: {str(e)}")
            print("Using default configuration...")
            return default_expert_features, 2
                
        # If no valid configuration found, use default
        if best_config is None:
            print("Warning: No valid configuration found. Using default configuration.")
            return default_expert_features, 2
            
        # Format final configuration
        expert_features = {}
        for i in range(len(best_config)):  # Use actual number of feature groups
            expert_features[f'expert_{i+1}'] = best_config[i]
            
        # Verify the configuration is valid
        if not expert_features or not all(expert_features.values()):
            print("Warning: Invalid optimized configuration. Using default configuration.")
            return default_expert_features, 2
            
        print("\nSelected feature configuration:")
        for expert_name, features in expert_features.items():
            print(f"{expert_name}: {features}")
            
        return expert_features, len(expert_features)
    
    def _assign_features_to_experts(self, features, n_experts):
        """Assign features to experts trying to balance information content"""
        if not features or n_experts <= 0:
            raise ValueError("Invalid features list or number of experts")
            
        # Ensure we have at least one feature per expert
        if len(features) < n_experts:
            n_experts = len(features)
            print(f"Warning: Reducing number of experts to {n_experts} due to feature count")
            
        # Simple initial implementation: distribute features evenly
        n_features = len(features)
        features_per_expert = max(1, n_features // n_experts)
        remainder = n_features % n_experts
        
        assignment = []
        start_idx = 0
        for i in range(n_experts):
            n_features_current = features_per_expert + (1 if i < remainder else 0)
            end_idx = start_idx + n_features_current
            if start_idx >= len(features):
                break
            current_features = features[start_idx:end_idx]
            if current_features:  # Only add non-empty feature sets
                assignment.append(current_features)
            start_idx = end_idx
            
        # Ensure we have at least one assignment
        if not assignment:
            assignment = [features]
            
        return assignment
    
    def _evaluate_config(self, feature_assignment):
        """Evaluate a particular expert configuration using cross-validation"""
        try:
            # Get training data
            train_data = next(iter(self.dataloaders['train']))[0].numpy()
            train_labels = next(iter(self.dataloaders['train']))[1].numpy()
            
            # Prepare cross-validation
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(train_data):
                X_train, X_val = train_data[train_idx], train_data[val_idx]
                y_train, y_val = train_labels[train_idx], train_labels[val_idx]
                
                # Evaluate each expert
                expert_scores = []
                for expert_features in feature_assignment:
                    # Get feature indices
                    feature_indices = [self.all_features.index(f) for f in expert_features]
                    
                    # Extract relevant features
                    X_train_expert = X_train[:, feature_indices]
                    X_val_expert = X_val[:, feature_indices]
                    
                    # Evaluate expert
                    score = self.evaluate_expert(
                        expert_features,
                        X_train_expert,
                        y_train,
                        X_val_expert,
                        y_val
                    )
                    expert_scores.append(score)
                
                # Combine expert scores (use mean for now)
                scores.append(np.mean(expert_scores))
                
            return np.mean(scores)
            
        except Exception as e:
            print(f"Warning: Error in configuration evaluation: {str(e)}")
            return 0.0  # Return lowest possible score on error