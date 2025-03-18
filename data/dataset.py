from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data_processor import Dataprocessor
from torch.utils.data import DataLoader

class TensorData(Dataset):
    """Dataset class for handling the Alzheimer's data"""
    def __init__(self, x_data, y_data, ids):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.ids = ids
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.ids[index]
    
    def __len__(self):
        return self.len
    
class AlzheimerDataset:
    """Class for handling Alzheimer's dataset loading and preprocessing"""

    def __init__(self, data_path, interesting_features, labels=['CN', 'AD'], sex='A', test = False, mean = None, std = None, common_ids = None):
        """Initialize dataset with configuration"""
        self.data_path = data_path
        self.interesting_features = interesting_features
        self.labels = labels
        self.sex = sex
        self.test = test
        self.mean = mean
        self.std = std
        self.common_ids = common_ids

        self.load_and_split_data()
        
    def load_and_split_data(self):
        """Load data and split into train, validation and test sets"""
        # Load data
        df = pd.read_csv(self.data_path)

        if self.common_ids is not None:
            df = df[df['id'].isin(self.common_ids)]

        # Extract relevant columns
        S = df['Sex'].values
        X = df[[str(i) for i in self.interesting_features]].values
        Y = df['label'].values
        ids = df['id'].values

        # Process data
        X, Y, ids = Dataprocessor.extract_label(S, X, Y, ids, labels = self.labels, sex = self.sex)


        if self.test:
            # For test data, use provided mean and std
            X = Dataprocessor.gaussian2(X, self.mean, self.std)
            self.X_test = X
            self.Y_test = Y
            return {
                'test': (X, Y, ids)
        }
        
        else:
            # For training data, compute mean and std
            X, mean, std = Dataprocessor.gaussian(X)
            self.mean = mean
            self.std = std
            #################형진 코드 추가_데이터 세트1:1###################
            mask_0 = (Y == 0)
            mask_1 = (Y == 1)

            X_0, X_1 = X[mask_0], X[mask_1]
            Y_0, Y_1 = Y[mask_0], Y[mask_1]
            ids_0, ids_1 = ids[mask_0], ids[mask_1]

            test_count = min(int(0.3 * len(X_0)), int(0.3 * len(X_1)))
            print(len(X_0)/test_count)

            X_trainval0, X_test0, Y_trainval0, Y_test0, ids_trainval0, ids_test0 = train_test_split(
                X_0, Y_0, ids_0, test_size=test_count/len(X_0), shuffle=False, random_state=42)
            
            # MCI 클래스에서 샘플링
            X_trainval1, X_test1, Y_trainval1, Y_test1, ids_trainval1, ids_test1 = train_test_split(
                X_1, Y_1, ids_1, test_size=test_count/len(X_1), shuffle=False, random_state=42)
            
            X_test = np.concatenate([X_test0, X_test1], axis=0)
            Y_test = np.concatenate([Y_test0, Y_test1], axis=0)
            ids_test = np.concatenate([ids_test0, ids_test1], axis=0)
            
            # 훈련 데이터 결합
            X_train_val = np.concatenate([X_trainval0, X_trainval1], axis=0)
            Y_train_val = np.concatenate([Y_trainval0, Y_trainval1], axis=0)
            ids_train_val = np.concatenate([ids_trainval0, ids_trainval1], axis=0)

            X_train, X_val, Y_train, Y_val, ids_train, ids_val = train_test_split(
                X_train_val, Y_train_val, ids_train_val, test_size=0.1, shuffle=False
            )
            
            return {
                'train': (X_train, Y_train, ids_train),
                'val': (X_val, Y_val, ids_val),
                'test': (X_test, Y_test, ids_test)
            }
    
    def data_loaders(self, batch_size=32):
        """Create DataLoader objects for all splits"""
        data_splits = self.load_and_split_data()
        g = torch.Generator()
        g.manual_seed(777)
        
        dataloaders = {}
        for split_name, (X, Y, ids) in data_splits.items():

            dataset = TensorData(X, Y, ids)
            shuffle = split_name == 'train'  
            dataloaders[split_name] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
            )
            
        return dataloaders
    