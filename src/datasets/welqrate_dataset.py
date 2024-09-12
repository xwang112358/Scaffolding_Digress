from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
    
atom_decoder = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'] # 'As', 'B', 'Se', 'Hg', 'Si', 'I', 'P',

class WelQrateDataset(InMemoryDataset):
    
    def __init__(self, stage, root, filter_dataset: bool, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['train_welqrate.csv', 'val_welqrate.csv', 'test_welqrate.csv']
        
    @property
    def split_file_name(self):
        return ['train_welqrate.csv', 'val_welqrate.csv', 'test_welqrate.csv']
    
    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'val_filtered.pt', 'test_filtered.pt']
        else:
            return ['train.pt', 'valid.pt', 'test.pt']

    
    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)['SMILES'].values

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list)):
            invalid = False
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                try:
                    type_idx.append(types[atom.GetSymbol()])
                except KeyError:
                    print(f"Atom {atom.GetSymbol()} not in atom_decoder, skipping molecule")
                    invalid = True
            
            if invalid:
                continue

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.stage}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")

class WelQrateDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = True 
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': WelQrateDataset(stage='train', root=root_path, filter_dataset=self.filter_dataset),
                    'val': WelQrateDataset(stage='val', root=root_path, filter_dataset=self.filter_dataset),
                    'test': WelQrateDataset(stage='test', root=root_path, filter_dataset=self.filter_dataset)}
        super().__init__(cfg, datasets)

class WelQrateinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = 'WelQrate'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19, 5: 28, 6: 31, 7: 32, 8: 35.4, 9: 79.9, 10: 127}
        self.valencies = [1, 4, 3, 2, 1, 4, 5, 2, 1, 1, 1]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 766 # to be computed

        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')

        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4691e-06,
        7.4074e-06, 7.4074e-06, 1.7284e-05, 7.6543e-05, 3.5556e-04, 1.4247e-03,
        2.8321e-03, 5.0469e-03, 8.7951e-03, 1.4435e-02, 2.1230e-02, 3.0072e-02,
        3.8175e-02, 4.7284e-02, 5.5975e-02, 6.2704e-02, 6.7338e-02, 7.0894e-02,
        7.2486e-02, 7.0533e-02, 6.8030e-02, 6.3973e-02, 5.8835e-02, 5.2119e-02,
        4.5244e-02, 3.7919e-02, 3.1701e-02, 2.4467e-02, 1.7588e-02, 1.2326e-02,
        7.3358e-03, 3.7235e-03, 2.3556e-03, 1.5012e-03, 1.0519e-03, 7.0123e-04,
        5.7531e-04, 3.3086e-04, 1.4321e-04, 7.6543e-05, 8.6420e-05, 5.9259e-05,
        3.2099e-05, 2.9630e-05, 2.4691e-05, 9.8765e-06, 2.7160e-05, 1.2346e-05,
        1.2346e-05, 1.2346e-05, 7.4074e-06])
        
        self.max_n_nodes = len(self.n_nodes) - 1
        
        #['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
        
        self.node_types = torch.tensor([5.6293e-07, 7.2313e-01, 1.1861e-01, 1.1945e-01, 8.2014e-03, 5.6293e-07,
        1.5267e-04, 2.2412e-02, 6.3035e-03, 1.7209e-03, 9.5698e-06])
        self.edge_types = torch.tensor([9.1277e-01, 3.9186e-02, 6.6274e-03, 1.6649e-04, 4.1254e-02])
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0000, 0.0897, 0.2800, 0.3603, 0.2540, 0.0085, 0.0075])
        
        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies
        # after we can be sure we have the data, complete infos
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)





def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_path = os.path.join(base_path, cfg.dataset.datadir)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles




if __name__ == "__main__":
    ds = [WelQrateDataset(s, os.path.join(os.path.abspath(__file__), "../../../data/welqrate"),
                       filter_dataset=False) for s in ["train", "val", "test"]]


