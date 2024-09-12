import graph_tool as gt
import os
import pathlib
import warnings
from tqdm import tqdm
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch_geometric.loader import DataLoader
from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from tqdm import tqdm
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from datasets.welqrate_dataset import WelQrateDataset

warnings.filterwarnings("ignore", category=PossibleUserWarning)

weight_dict = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19, 5: 28, 6: 31, 7: 32, 8: 35.4, 9: 79.9, 10: 127}

def calculate_max_molecule_weight(train_dataset):
    max_weight = 0

    # Wrap the dataset iteration with tqdm
    for data in tqdm(train_dataset, desc="Processing molecules", unit="molecule"):
        # Convert one-hot vectors to indices
        molecule_types = torch.argmax(data.x, dim=1)
        
        # Calculate the total weight of the molecule
        molecule_weight = sum(weight_dict[int(atom_type)] for atom_type in molecule_types)
        
        # Update max_weight if this molecule is heavier
        max_weight = max(max_weight, molecule_weight)

    return max_weight

    

if __name__ == '__main__':
    ds = [WelQrateDataset(s, os.path.join(os.path.abspath(__file__), "../../data/welqrate"),
                       filter_dataset=False) for s in ["train", "val", "test"]]
    
    train_dataset = ds[0]
    
    problem_indices = [4507, 62287, 66149, 75884, 85563, 94299, 
                       101401, 124721, 127034, 147746, 151050, 
                       158602, 171739, 172358, 178779, 183586, 
                       194416, 196035, 196628, 217607, 246562, 
                       260694, 282642, 286194, 324631, 326739, 
                       341800, 344731, 357199, 357855]
    
    for i in problem_indices:
        # check number of atoms and max edge index value
        number_of_atoms = train_dataset[i].x.shape[0]
        max_edge_index = train_dataset[i].edge_index.max().item()
        min_edge_index = train_dataset[i].edge_index.min().item()
        
        print(f"Number of atoms: {number_of_atoms}")
        print(f"Max edge index: {max_edge_index}")
        print(f"Min edge index: {min_edge_index}")
        
        

    
    
    
    # train_max_weight = calculate_max_molecule_weight(train_dataset)
    # print(f"The maximum weight of molecules in the dataset is: {train_max_weight}")
    
    
    # val_max_weight = calculate_max_molecule_weight(val_dataset)
    # print(f"The maximum weight of molecules in the validation dataset is: {val_max_weight}")
    # count = 0
    # indices = []
    # error_message = []
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # for i, batch in enumerate(tqdm(train_loader)):
    #     if batch.edge_index.numel() == 0:
    #         raise ValueError("Empty edge in batch")

    #     try:
    #         dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    #         dense_data = dense_data.mask(node_mask)
    #     except:
    #         count += 1
    #         indices.append(i)
    #         continue

    # print(count)
    # print(indices)

    