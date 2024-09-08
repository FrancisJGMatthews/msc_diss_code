import pandas as pd 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda


# ----------------------- DATA LOADING ----------------------- #

def trainingDataRead(filepath):
    """
    Load CSV of following samples and return variables needed to load into custom PyTorch data class. 

    Args:
        - filepath (str) : filepath to edgelist CSV file containing rows of congress -> user following relationships
    Returns:
        - tuples_list (list(tuple)) : list of tuples. Each tuple consists of (congress, user) twitter username
        - feature_index_map (dict) : dictionary of structure {username : int} assigning each congress username to a unique integer
        - label_index_map (dict) : dictionaryof structure {username : int} assigning each followee username to a unique interger
    """
    edgelist = pd.read_csv(filepath, header=None)
    samples = list(edgelist.itertuples(index=False, name=None))

    print(f'{len(samples)} samples')

    # Get lists of features, labels
    house_features = list(set(i[0] for i in samples))
    house_labels = list(set(i[1] for i in samples))

    print(f'{len(house_features)} features and {len(house_labels)} labels')

    # Generate label_to_index dictionary (maps each user to a unique number)
    feature_index_map = {label: idx for idx, label in enumerate(house_features)}
    label_index_map = {label: idx for idx, label in enumerate(house_labels)}

    return samples, feature_index_map, label_index_map



# ----------------------- TRAINING DATA CLASS ----------------------- #

# Custom dataset class for training data. Training data to be passed as a list of tuples (congressmember, followee) with transform/target_transform defined

class FollowingDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample_data = self.data[index]
        features = sample_data[0]
        label = sample_data[1]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features, label
    


# ----------------------- NETWORK TRAINING ----------------------- #

    # Function to perform the training for a single epoch
def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    size = len(dataloader.dataset) # Get number of samples
    model.train()

    epoch_losses = []
    for batch, (X, y) in enumerate(dataloader):
        
        # Move X,y data to device
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print loss information
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            epoch_losses.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return epoch_losses