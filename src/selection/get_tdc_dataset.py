import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch
from selection.helper import _generate_scaffold, generate_scaffolds_dict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np
from tdc.single_pred import HTS
from selection.data_utils import smiles2graph


adme_list = ['PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 
                'Bioavailability_Ma', 'BBB_Martins', 'CYP2C19_Veith',
                'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith',
                'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
                'CYP3A4_Substrate_CarbonMangels']
tox_list = ['hERG', 'hERG_Karim', 'AMES', 'DILI', 'Skin Reaction',
            'Carcinogens_Lagunin', 'ClinTox', 'Tox21']
hts_list = ['HIV', 'SARSCoV2_Vitro_Touret']
    


def get_tdc_dataset(name, path, split_method = 'scaffold', label_name = None):
    
    def get_data_list(smiles_list, y_list, type = 'train'):
        
        try:
            data_list = torch.load(os.path.join(path, f'{type}_pyg_data_list.pt'))
            return data_list
        except:
            data_list = []
            for i in tqdm(range(len(smiles_list))):
                smiles = smiles_list[i]
                # print(smiles)
                y = y_list[i]
                pyg_data = smiles2graph(smiles)
                
                if pyg_data is None:
                    raise ValueError('Error in converting smiles to graph')
                
                pyg_data.y = torch.tensor([y], dtype=torch.float)
                # print(pyg_data)
                data_list.append(pyg_data)    
            # save the data_list
            torch.save(data_list, os.path.join(path, f'{type}_pyg_data_list.pt'))
        
        return data_list
    
    path = os.path.join(path, name)

    if name in adme_list:
        from tdc.single_pred import ADME
        data = ADME(path = path, name = name)
    elif name in tox_list:
        from tdc.single_pred import Tox
        data = Tox(path = path, name = name, label_name = label_name)
    elif name in hts_list:
        data = HTS(path = path, name = name)
    
    split_dict = data.get_split(method=split_method)
    
    data_dict = {}
    
    for key in split_dict.keys():
        set_name = key
        print(f'Processing {set_name} set')
        df = split_dict[key]
        smiles_list = df['Drug']
        y_list = df['Y']
        data_list = get_data_list(smiles_list, y_list, type = set_name)
        data_dict[set_name] = data_list

        # data_list = torch.load(os.path.join(path, f'{set_name}_pyg_data_list.pt'))

        
        
    # gather the size distribution of the dataset
    # size_list = []
    # for data in data_list:
    #     size_list.append(data.num_nodes)
    # print('The size distribution of the dataset is: ', np.unique(size_list, return_counts=True))
    
    return data_dict 
    
    