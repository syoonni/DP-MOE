import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Dataprocessor:
    """Class for data preprocessing operations"""

    @staticmethod
    def extract_limit(data, low=0.005, high=0.995):
        """Extract the lower and upper bounds for data clipping"""
        lower_bound = np.quantile(data, axis=1, q=low, method='nearest', keepdims=True)
        upper_bound = np.quantile(data, axis=1, q=high, method='nearest', keepdims=True)
        return lower_bound, upper_bound
    
    def extract_label(S, X, Y, ids, labels=['CN', 'AD'], sex = 'A'): 
        """Extract and process labels"""
        X2 = []
        Y2 = []
        ids2 = []

        if sex == 'A': # use all sex data
            for i in range(len(Y)):
                if Y[i] in labels:
                    X2.append(X[i]) 
                    ids2.append(ids[i])
                    # label mapping
                    Y2.append(labels.index(Y[i]))

        else: # use only the specified
            for i in range(len(Y)):
                if S[i] == sex:
                    X2.append(X[i])
                    ids2.append(ids[i])
                    # label mapping
                    Y2.append(labels.index(Y[i]))
            
        # Convert to numpy arrays
        X2 = np.array(X2).astype(float)
        Y2 = np.array(Y2)
        ids2 = np.array(ids2)

        # Print statistics
        print(f"Total samples: {len(Y2)}")
        print(f"{labels[1]}: {sum(Y2)}")
        print(f"{labels[0]}: {len(Y2) - sum(Y2)}")

        return X2, Y2, ids2 
    
    @staticmethod
    def gaussian(X): 
        """Gaussian normalization with outlier clipping"""
        X = X.T
        Zmin, Zmax = Dataprocessor.extract_limit(X)
        mean = np.expand_dims(np.mean(X, axis=1), axis=1)
        std = np.expand_dims(np.std(X, axis=1), axis=1)
        X = np.clip(X, Zmin, Zmax)
        X = (X - mean) / std
        X = X.T
        return X, mean, std
    
    @staticmethod
    def gaussian2(X, mean, std):
        """Apply pre-computed gaussian noramlization"""
        X = X.T
        Zmin, Zmax = Dataprocessor.extract_limit(X)
        X = np.clip(X, Zmin, Zmax)
        X = (X - mean) / std
        X = X.T
        return X 

    def add_noise(data, noise_factor=0.1):
        """Add Gaussian noise to the input data"""
        noise = noise_factor * torch.randn_like(data)
        data[:, 1:] = data[:, 1:] + noise[:, 1:]
        return data

