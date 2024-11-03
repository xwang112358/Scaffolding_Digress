
from ogb.utils.features import (atom_to_feature_vector,bond_to_feature_vector) 

import torch
import numpy as np

import copy
import pathlib
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

from selection.mol_utils import bond_to_feature_vector as bond_to_feature_vector_non_santize
from selection.mol_utils import atom_to_feature_vector as atom_to_feature_vector_non_santize

from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges


# smiles2graph used in WelQrate
def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to 2D graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        mol = mol if removeHs else Chem.AddHs(mol)
        if reorder_atoms:
            mol, _ = ReorderCanonicalRankAtoms(mol)

    except Exception as e:
        print(f'cannot generate mol, error: {e}, smiles: {smiles_string}')

    if mol is None:

        smiles = smiles_cleaner(smiles_string)
        try:
            mol = Chem.MolFromSmiles(smiles)

            mol = mol if removeHs else Chem.AddHs(mol)
            if reorder_atoms:
                mol, _ = ReorderCanonicalRankAtoms(mol)

        except Exception as e:
            print(f'cannot generate mol, error: {e}, smiles: {smiles_string}')
            mol = None

        if mol is None:
            raise ValueError(f'cannot generate molecule with smiles: {smiles_string}')

    else:
        # calculate Gasteiger charges
        rdPartialCharges.ComputeGasteigerCharges(mol)
        
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))

        atom_features_list = atomized_mol_level_features(atom_features_list, mol)
        x = torch.tensor(atom_features_list, dtype=torch.float32)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                bond_attr = []
                bond_attr += one_hot_vector(bond.GetBondTypeAsDouble(),
                                            [1.0, 1.5, 2.0, 3.0])
                is_aromatic = bond.GetIsAromatic()
                is_conjugated = bond.GetIsConjugated()
                is_in_ring = bond.IsInRing()
                bond_attr.append(is_aromatic)
                bond_attr.append(is_conjugated)
                bond_attr.append(is_in_ring)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(bond_attr)
                edges_list.append((j, i))
                edge_features_list.append(bond_attr)

            edge_index = torch.tensor(edges_list).t().contiguous()
            edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)

        else:  # mol has no bonds
            edge_index = torch.from_numpy(np.empty((2, 0)))
            edge_attr = torch.from_numpy(np.empty((0, num_bond_features)))
            print('Warning: molecule does not have bond:', smiles_string)

        graph = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            num_nodes=torch.tensor([len(x)]),
            num_edges=torch.tensor(len(edge_index[0])),
            smiles=smiles_string
                            )

    return graph

def atomized_mol_level_features(all_atom_features, mol):
    '''
    Get more atom features that cannot be calculated only with atom,
    but also with mol
    :param all_atom_features:
    :param mol:
    :return:
    '''
    # Crippen has two parts: first is logP, second is Molar Refactivity(MR)
    all_atom_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    all_atom_TPSA_contrib = rdMolDescriptors._CalcTPSAContribs(mol)
    all_atom_ASA_contrib = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    all_atom_EState = EState.EStateIndices(mol)

    new_all_atom_features = []
    for atom_id, feature in enumerate(all_atom_features):
        crippen_logP = all_atom_crippen[atom_id][0]
        crippen_MR = all_atom_crippen[atom_id][1]
        atom_TPSA_contrib = all_atom_TPSA_contrib[atom_id]
        atom_ASA_contrib = all_atom_ASA_contrib[atom_id]
        atom_EState = all_atom_EState[atom_id]

        feature.append(crippen_logP)
        feature.append(crippen_MR)
        feature.append(atom_TPSA_contrib)
        feature.append(atom_ASA_contrib)
        feature.append(atom_EState)

        new_all_atom_features.append(feature)
    return new_all_atom_features

def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order

def one_hot_vector(val, lst):
	'''
	Converts a value to a one-hot vector based on options in lst
	'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)


pattern_dict = {'[NH-]': '[N-]', '[OH2+]':'[O]'}
def smiles_cleaner(smiles):
    '''
    This function is to clean smiles for some known issues that makes
    rdkit:Chem.MolFromSmiles not working
    '''
    print('fixing smiles for rdkit...')
    new_smiles = smiles
    for pattern, replace_value in pattern_dict.items():
        if pattern in smiles:
            print('found pattern and fixed the smiles!')
            new_smiles = smiles.replace(pattern, replace_value)
    return new_smiles