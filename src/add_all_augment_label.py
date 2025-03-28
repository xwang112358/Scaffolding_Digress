# import graph_tool as gt
import os
import pathlib
import warnings
import time
import torch
from torch_geometric.loader import DataLoader
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import pandas as pd
from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from torch.nn import functional as F
from src.diffusion_model_discrete_scaffold import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from selection.augmentation import AugmentationDataset

warnings.filterwarnings("ignore", category=PossibleUserWarning)

atom_decoder = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

def convert_graphs(molecule_list, label_list):
    """Convert the generated graphs to pyg graphs"""
    from torch_geometric.data import Data
    
    pyg_graphs = []
    for i, mol in enumerate(molecule_list):    
        atom_types, edge_types = mol
        x = F.one_hot(atom_types, num_classes=len(atom_decoder))
        # Add an extra column to x
        x = torch.cat([x, torch.zeros(x.size(0), 1)], dim=1)
        edge_index = torch.nonzero(edge_types > 0, as_tuple=True)
        edge_index = torch.stack(edge_index, dim=0)  # Convert to (2, num_edges) format
        # Get edge attributes and convert to 0-based indexing
        edge_attr = edge_types[edge_index[0], edge_index[1]] - 1  # Subtract 1 to convert to 0-based indexing
        edge_attr = F.one_hot(edge_attr.long(), num_classes=4)
        # Switch edge attributes for double bond (index 1) and aromatic bond (index 3)
        mask_double = torch.all(edge_attr == torch.tensor([0,1,0,0]), dim=1)
        mask_aromatic = torch.all(edge_attr == torch.tensor([0,0,0,1]), dim=1)
        
        # Create new tensor with swapped values
        edge_attr_new = edge_attr.clone()
        edge_attr_new[mask_double] = torch.tensor([0,0,0,1])
        edge_attr_new[mask_aromatic] = torch.tensor([0,1,0,0])
        edge_attr = edge_attr_new
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y= torch.tensor([label_list[i]], dtype=torch.int32))
        pyg_graphs.append(data)
    return pyg_graphs


@hydra.main(version_base='1.3', config_path='../configs', config_name='welqrate')
def main(cfg: DictConfig):

    name = cfg.augment_data.name
    split_scheme = cfg.augment_data.split
    ratio = cfg.augment_data.ratio

    # load the generated graphs csv
    generated_graphs_path = f'./generated_graphs/{name}_{split_scheme}_{ratio}_generated_graphs.pt'
    generated_graphs = torch.load(generated_graphs_path)

    # load the sampled smiles
    sampled_smiles_path = f'./sampled_smiles/sampled_smiles_{name}_{split_scheme}_{ratio}.csv'
    sampled_smiles_df = pd.read_csv(sampled_smiles_path)
    sampled_scaffolds = sampled_smiles_df['scaffold'].tolist()
    sampled_smiles = sampled_smiles_df['smiles'].tolist()
    sampled_labels = sampled_smiles_df['y'].tolist()  
    print(f'sampled_smiles: {len(sampled_smiles)}')
    
    selected_data = AugmentationDataset(cfg = cfg, smiles_list = sampled_scaffolds, 
                                        label_list = sampled_labels)
    print(f'selected_data: {len(selected_data)}')

    label_list = []

    for data in selected_data:
        label_list.append(data.y.item())
    
    # print(label_list)
    # for i in range(len(generated_graphs)):
    #     generated_graphs[i].y = torch.tensor([label_list[i]], dtype=torch.int32)
    
    pyg_graphs = convert_graphs(generated_graphs, label_list)
    path = f'/home/allenwang/scaffold-aware/Scaffolding_Digress/augment_pyg_graphs_labels/{name}_{split_scheme}_{ratio}_augment_pyg_graphs_labels.pt'
    torch.save(pyg_graphs, path)

    


    



        

    
    



if __name__ == '__main__':
    main()
