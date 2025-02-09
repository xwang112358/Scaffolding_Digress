# from ogb.graphproppred import PygGraphPropPredDataset
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch.nn.functional as F
from selection.helper import _generate_scaffold, generate_scaffolds_dict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src import utils
from src.analysis.rdkit_functions import build_molecule_with_partial_charges, mol2smiles, build_molecule

# Add this line at the top of the file with other imports
__all__ = ['AugmentationDatasetSelector', 'SMILESRoundTripChecker', 'AugmentationDataset']

# def get_dataset(args, load_path, load_unlabeled_name="None"):
#     if load_unlabeled_name=='None':
#         if args.dataset.startswith('plym'):
#             return PolymerRegDataset(args.dataset, load_path)
#         elif args.dataset.startswith('ogbg'):
#             ogbg_dataset = PygGraphPropPredDataset(args.dataset, load_path)
#             label_split_idx = ogbg_dataset.get_idx_split()
#             meta_info = ogbg_dataset.meta_info
#             ogbg_data_list = [data for data in ogbg_dataset]
#             smile_path = os.path.join('./raw_data', '_'.join(args.dataset.split('-')), 'mapping/mol.csv.gz')
#             smiles = pd.read_csv(smile_path, compression='gzip', usecols=['smiles'])
#             smiles_list = smiles['smiles'].tolist()
            
#             new_dataset = augmentation_dataset(args.dataset, load_path, ogbg_data_list, smiles_list, label_split_idx, meta_info)
            
#             return new_dataset #PygGraphPropPredDataset(args.dataset, load_path)
#     else:
#         raise ValueError('Unlabeled dataset {} not supported'.format(load_unlabeled_name))
    

class AugmentationDatasetSelector:
    def __init__(self, name, root, smiles_list, y_list):
        self.name = name
        self.root = root
        self.smiles_list = smiles_list
        self.y_list = y_list
        self.scaff_list = [_generate_scaffold(smi) for smi in smiles_list]
        self.cluster_ids = None

    def scaffold_clustering(self, cutoff):
        print('extracting scaffolds')
        scaff_mols = [Chem.MolFromSmiles(scaffold) for scaffold in self.scaff_list]
        ecfps = []
        for mol in scaff_mols:
            if mol is None:
                raise ValueError('Invalid Scaffold SMILES. Please check.')
            try:
                ecfps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            except Exception as e:
                raise ValueError(f'Error generating Morgan fingerprint: {e}')
        
        print('calculating distance matrix')
        dists = calc_distance_matrix(ecfps)
        clusters = Butina.ClusterData(dists, len(ecfps), cutoff, isDistData=True)
        self.cluster_ids = [0] * len(ecfps)

        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                self.cluster_ids[idx] = cluster_id
        
        print(f'Clustered {len(ecfps)} training data into {len(clusters)} clusters.')
        
        return self.cluster_ids
    
    
    def augmentation_sampling(self, N, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        num_molecules = len(self.smiles_list)
        assert len(self.cluster_ids) == num_molecules

        cluster_ids = np.array(self.cluster_ids)
        labels = np.array(self.y_list)

        unique_clusters = np.unique(cluster_ids)
        unique_classes = np.unique(labels)
        print(f'Unique clusters: {len(unique_clusters)}, Unique classes: {len(unique_classes)}')

        N_s = {}
        for s in unique_clusters:
            N_s[s] = np.sum(cluster_ids == s)

        epsilon = 1e-6
        w_s = {}
        for s in unique_clusters:
            w_s[s] = 1.0 / (N_s[s] + epsilon)

        sum_w_s = sum(w_s.values())
        P_s = {s: w_s[s] / sum_w_s for s in unique_clusters}

        N_c = {}
        for c in unique_classes:
            N_c[c] = np.sum(labels == c)

        w_c = {}
        for c in unique_classes:
            w_c[c] = 1.0 / (N_c[c] + epsilon)

        sum_w_c = sum(w_c.values())
        P_c = {c: w_c[c] / sum_w_c for c in unique_classes}

        sampling_probs = np.zeros(num_molecules)
        for idx in range(num_molecules):
            s = cluster_ids[idx]
            c = labels[idx]
            P_i = P_s[s] * P_c[c]
            sampling_probs[idx] = P_i

        total_prob = np.sum(sampling_probs)
        sampling_probs /= total_prob

        print('Check Sum of all probabilities across all molecules:', sum(sampling_probs))

        sampled_indices = np.random.choice(num_molecules, size=N, replace=True, p=sampling_probs)
        
        selected_smiles = [self.smiles_list[idx] for idx in sampled_indices]
        selected_labels = [self.y_list[idx] for idx in sampled_indices]
        selected_scaffolds = [self.scaff_list[idx] for idx in sampled_indices]
        selected_data = pd.DataFrame({'smiles': selected_smiles, 'scaffold': selected_scaffolds, 'y': selected_labels})
        # create directory
        
        # selected_data.to_csv(os.path.join(self.root, self.name, f'{self.name}_selected_train_data.csv'), index=False)
        
        return selected_data
    
    def plot(self, save_path):
        pass


# ----------------- AugmentationDataset -----------------
atom_decoder = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']


class AugmentationDataset(InMemoryDataset):
    def __init__(self, cfg, smiles_list, label_list, filter_dataset = False, transform=None, pre_transform=None):
        name = cfg.augment_data.name
        root = cfg.augment_data.data_dir
        self.split_scheme = cfg.augment_data.split
        self.ratio = cfg.augment_data.ratio
        self.new_label_list = []
        self.name = '_'.join(name.split('-'))
        self.root = root
        self.smiles_list = smiles_list
        self.label_list = label_list
        self.total_data_len = len(smiles_list)
        self.filter_dataset = filter_dataset
        self.atom_decoder = atom_decoder
    
        super(AugmentationDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'selected_processed')
        
    @property
    def processed_file_names(self):
        return [f'{self.name}_{self.split_scheme}_{self.ratio}_selected_data.pt']

    def process(self):
        # define atom_decoder later
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        data_list = []
        smiles_kept = []
        
        for i, smile in enumerate(tqdm(self.smiles_list)):
            invalid = False
            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                smile = smiles_cleaner(smile)
                mol = Chem.MolFromSmiles(smile) 

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                raise ValueError(f"Cannot retrieve scaffold for {smile}")

            scaffold_indices = mol.GetSubstructMatch(scaffold)
            if not scaffold_indices:
                print(f"Cannot find matched substructure in {smile}")
                N = mol.GetNumAtoms()
                scaffold_indices = [i for i in range(N)]
                # raise ValueError(f"Cannot find matched substructure in {smile}")
                
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
            
            # scaffold node mask
            node_mask = torch.zeros(N, dtype=torch.bool)
            for idx in scaffold_indices:
                node_mask[idx] = True
            
            row, col, edge_type = [], [], []
            # edge_mask = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]
                
            if len(row) == 0:
                print("No bonds found, skipping molecule")
                continue
            
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)
            
            # Sorting to ensure consistent order
            perm = (edge_index[0] * N + edge_index[1]).argsort() 
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
            # edge_mask = torch.tensor(edge_mask, dtype=torch.bool)[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)
            self.new_label_list.append(self.label_list[i])
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, 
                        node_mask=node_mask)
            
            smiles_kept.append(smile)
            data_list.append(data)
            # save the kept smiles
        torch.save(smiles_kept, os.path.join(self.root, self.name, f'{self.name}_{self.split_scheme}_{self.ratio}_selected_filtered_smiles.pt'))

        torch.save(self.collate(data_list), self.processed_paths[0])
    
            # if self.filter_dataset:
            #     smiles_save_path = os.path.join(self.root, self.name, f'{self.name}_selected_augment.smiles') 
            #     print(smiles_save_path)
            #     with open(smiles_save_path, 'w') as f:
            #         f.writelines('%s\n' % s for s in smiles_kept)
                    
            #     print(f"Number of molecules kept: {len(smiles_kept)} / {len(self.smiles_list)}")
    def get_label_list(self):
        return self.new_label_list

           
# class SMILESRoundTripChecker:
#     def __init__(self, smiles_list, atom_decoder=atom_decoder):
#         self.smiles_list = smiles_list
#         self.atom_decoder = atom_decoder
#         self.valid_smiles = []
#         self.invalid_smiles = []
#         self.conversion_errors = []
        
#     def check_round_trip(self):
#         """Check SMILES that can successfully round-trip through molecular graph conversion"""
#         types = {atom: i for i, atom in enumerate(self.atom_decoder)}
#         bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        
#         for smile in tqdm(self.smiles_list, desc="Checking SMILES round-trip conversion"):
#             error_info = {'smiles': smile, 'error': None, 'stage': None}
            
#             try:
#                 # 1. Initial SMILES to Mol conversion
#                 print('converting mol to molecular graph...')
#                 mol = Chem.MolFromSmiles(smile,)
#                 if mol is None:
#                     smile = smiles_cleaner(smile)
#                     mol = Chem.MolFromSmiles(smile)
#                     if mol is None:
#                         error_info['error'] = 'Invalid SMILES'
#                         error_info['stage'] = 'initial_conversion'
#                         self.conversion_errors.append(error_info)
#                         self.invalid_smiles.append(smile)
#                         continue
                
#                 # 2. Convert to molecular graph representation
#                 N = mol.GetNumAtoms()
#                 type_idx = []
#                 for atom in mol.GetAtoms():
#                     if atom.GetSymbol() not in types:
#                         error_info['error'] = f'Unknown atom type: {atom.GetSymbol()}'
#                         error_info['stage'] = 'atom_type_check'
#                         self.conversion_errors.append(error_info)
#                         self.invalid_smiles.append(smile)
#                         continue
#                     type_idx.append(types[atom.GetSymbol()])
                
#                 row, col, edge_type = [], [], []
#                 for bond in mol.GetBonds():
#                     start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#                     row += [start, end]
#                     col += [end, start]
#                     edge_type += 2 * [bonds[bond.GetBondType()] + 1]
                
#                 if len(row) == 0:
#                     error_info['error'] = 'No bonds found'
#                     error_info['stage'] = 'bond_processing'
#                     self.conversion_errors.append(error_info)
#                     self.invalid_smiles.append(smile)
#                     continue
                
#                 # 3. Create PyG Data object
#                 edge_index = torch.tensor([row, col], dtype=torch.long)
#                 edge_type = torch.tensor(edge_type, dtype=torch.long)
#                 edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)
#                 x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
#                 data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                
#                 # 4. Convert to dense representation
#                 dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, None)
#                 dense_data = dense_data.mask(node_mask, collapse=True)
#                 X, E = dense_data.X, dense_data.E
                
#                 # 5. Try to rebuild molecule
#                 print('rebuilding molecule...')
#                 rebuilt_mol = build_molecule(X[0], E[0], self.atom_decoder)
#                 rebuilt_smiles = mol2smiles(rebuilt_mol)
                
#                 if rebuilt_smiles is None:
#                     error_info['error'] = 'Failed to convert rebuilt molecule to SMILES'
#                     error_info['stage'] = 'final_conversion'
#                     self.conversion_errors.append(error_info)
#                     self.invalid_smiles.append(smile)
#                 else:
#                     self.valid_smiles.append(smile)
                    
#             except Exception as e:
#                 error_info['error'] = str(e)
#                 error_info['stage'] = 'unexpected_error'
#                 self.conversion_errors.append(error_info)
#                 self.invalid_smiles.append(smile)
                
#         print(f"\nResults:")
#         print(f"Total SMILES processed: {len(self.smiles_list)}")
#         print(f"Valid conversions: {len(self.valid_smiles)}")
#         print(f"Invalid conversions: {len(self.invalid_smiles)}")
        
#         return self.valid_smiles, self.invalid_smiles, self.conversion_errors
    
#     def save_results(self, save_dir):
#         """Save the results to files"""
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Save valid SMILES
#         with open(os.path.join(save_dir, 'valid_smiles.txt'), 'w') as f:
#             f.writelines('%s\n' % s for s in self.valid_smiles)
            
#         # Save invalid SMILES
#         with open(os.path.join(save_dir, 'invalid_smiles.txt'), 'w') as f:
#             f.writelines('%s\n' % s for s in self.invalid_smiles)
            
#         # Save detailed error information
#         df = pd.DataFrame(self.conversion_errors)
#         df.to_csv(os.path.join(save_dir, 'conversion_errors.csv'), index=False)
        
#         print(f"\nResults saved to: {save_dir}")



                        
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
            




# convert the selected smiles to the pyg data object for the DiGress 

# class AugmentationDataset(InMemoryDataset):
#     def __init__(self, name, root, data_list, smile_list, split_dict, meta_dict=None,
#                  transform=None, pre_transform=None):
#         self.name = '_'.join(name.split('-'))
#         self.root = root
#         self.smile_list = smile_list
#         self.total_data_len = len(data_list)
#         self.data_list = data_list
#         self.meta_info = meta_dict
#         self.split_dict = split_dict
#         self.scaff_list = [_generate_scaffold(smi) for smi in smile_list]
        

#         super(AugmentationDataset, self).__init__(root, transform, pre_transform)
        
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         # Return an empty list since raw files are not used
#         return []

#     @property
#     def processed_file_names(self):
#         # Define the name of the processed file
#         return [f'{self.name}_scaff_processed.pt']

#     @property
#     def processed_dir(self):
#         # Override to save processed files directly in the root directory
#         return os.path.join(self.root, self.name, 'processed')
    

#     def process(self):
#         # Implement your processing logic here
#         # _, self.scaffold_sets = generate_scaffolds_dict(self.smile_list)
#         print(self.data_list[0])
#         # self.scaff_list = []
#         for i in tqdm(range(len(self.data_list))):
#             self.data_list[i].smiles = self.smile_list[i]
#             # scaff_smiles = _generate_scaffold(self.smile_list[i])
#             # self.scaff_list.append(scaff_smiles)
#             self.data_list[i].scaff_smiles = self.scaff_list[i]

#         print(self.data_list[:10])
#         print(len(self.data_list))
#         data, slices = self.collate(self.data_list)
#         # print(data)
#         # print(slices)
#         # Save the processed data to the root directory
#         torch.save((data, slices), self.processed_paths[0])
        
#         # self.data, self.slices = torch.load(self.processed_paths[0])
#     def augmentation_sampling(self, N, seed=42, plot=True, plot_path=None):
#         # Return the indices of the sampled, training data

#         # Set random seed for reproducibility
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#         train_idx = self.split_dict['train']
#         num_molecules = len(train_idx)

#         # Get cluster IDs and labels for training molecules
#         assert len(self.cluster_ids) == num_molecules
        
#         cluster_ids = np.array(self.cluster_ids)
#         labels = np.array([self.data_list[i].y.item() for i in train_idx])

#         unique_clusters = np.unique(cluster_ids)
#         unique_classes = np.unique(labels)
#         print(f'Unique clusters: {len(unique_clusters)}, Unique classes: {len(unique_classes)}')

#         # Step 1: Compute N_s (number of molecules in each scaffold cluster)
#         N_s = {}
#         for s in unique_clusters:
#             N_s[s] = np.sum(cluster_ids == s)

#         # Step 2: Compute w_s (inverse frequency weights for scaffold clusters)
#         epsilon = 1e-6
#         w_s = {}
#         for s in unique_clusters:
#             w_s[s] = 1.0 / (N_s[s] + epsilon)

#         # Normalize w_s to get P_s (sampling probability for scaffold clusters)
#         sum_w_s = sum(w_s.values())
#         P_s = {s: w_s[s] / sum_w_s for s in unique_clusters}

#         # Step 3: Compute N_c (global number of molecules for each class)
#         N_c = {}
#         for c in unique_classes:
#             N_c[c] = np.sum(labels == c)

#         # Compute w_c (inverse frequency weights for classes)
#         w_c = {}
#         for c in unique_classes:
#             w_c[c] = 1.0 / (N_c[c] + epsilon)

#         # Normalize w_c to get P_c (sampling probability for classes)
#         sum_w_c = sum(w_c.values())
#         P_c = {c: w_c[c] / sum_w_c for c in unique_classes}

#         # Step 4: Compute sampling probability P_i for each molecule
#         sampling_probs = np.zeros(num_molecules)
#         for idx in range(num_molecules):
#             s = cluster_ids[idx]
#             c = labels[idx]
#             P_i = P_s[s] * P_c[c]
#             sampling_probs[idx] = P_i

#         # Step 5: Normalize sampling probabilities so they sum to 1
#         total_prob = np.sum(sampling_probs)
#         sampling_probs /= total_prob

#         print('Sum of all probabilities across all molecules:', sum(sampling_probs))

#         # Step 6: Sample N molecules according to sampling_probs
#         sampled_indices = np.random.choice(num_molecules, size=N, replace=False, p=sampling_probs)
#         sampled_data_indices = [train_idx[idx].item() for idx in sampled_indices]

#         # Plotting (Optional)
#         if plot and plot_path is not None:
#             dir_path = os.path.dirname(plot_path)
#             os.makedirs(dir_path, exist_ok=True)

#             # Prepare data for plotting
#             cluster_class_counts = {}
#             for s in unique_clusters:
#                 cluster_class_counts[s] = {}
#                 for c in unique_classes:
#                     # Count molecules in cluster s with class c
#                     idx_s_c = np.where((cluster_ids == s) & (labels == c))[0]
#                     cluster_class_counts[s][c] = len(idx_s_c)

#             # Prepare data for stacked bar plot
#             clusters = sorted(unique_clusters)
#             classes = sorted(unique_classes)
#             counts_per_class = {c: [] for c in classes}
#             for s in clusters:
#                 for c in classes:
#                     counts_per_class[c].append(cluster_class_counts[s][c])

#             # Plotting
#             bar_width = 0.8
#             fig, ax = plt.subplots(figsize=(12, 6))

#             bottom = np.zeros(len(clusters))
#             for c in classes:
#                 ax.bar(clusters, counts_per_class[c], bottom=bottom, width=bar_width, label=f'Class {c}')
#                 bottom += counts_per_class[c]

#             ax.set_xlabel('Scaffold Cluster ID')
#             ax.set_ylabel('Count')
#             ax.set_title('Class Distribution in Each Scaffold Cluster')
#             ax.legend(title='Class')
#             ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#             plt.savefig(plot_path)
#             plt.close(fig)

#         return sampled_data_indices


    # def augmentation_sampling(self, N, seed=42, plot=True, plot_path=None):
    #     # return the indices of the sampled, training data
        
    #     # calculate the joint distribution of the scaffold and the label for training data
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
        
    #     train_idx = self.split_dict['train']
    #     num_molecules = len(self.cluster_ids)
        
    #     # Get cluster IDs and labels for training molecules
    #     cluster_ids = np.array(self.cluster_ids)
    #     print(cluster_ids)
    #     labels = np.array([self.data_list[i].y.item() for i in train_idx])
        
    #     unique_clusters = np.unique(cluster_ids)
    #     unique_classes = np.unique(labels)
    #     print(f'Unique clusters: {len(unique_clusters)}, Unique classes: {len(unique_classes)}')
        
    #     # Step 1: Compute N_s (number of molecules in each scaffold cluster)
    #     N_s = {}
    #     for s in unique_clusters:
    #         N_s[s] = np.sum(cluster_ids == s)
        
    #     # Step 2: Compute w_s (inverse frequency weights for scaffold clusters)
    #     epsilon = 1e-6
    #     w_s = {}
    #     for s in unique_clusters:
    #         w_s[s] = 1.0 / (N_s[s] + epsilon)
        
    #     # Normalize w_s to get P_s (sampling probability for scaffold clusters)
    #     sum_w_s = sum(w_s.values())
    #     P_s = {s: w_s[s] / sum_w_s for s in unique_clusters}
        
    #     # Step 3: Compute N_{s,c} (number of molecules for each class within each scaffold cluster)
    #     N_sc = {}
    #     w_sc = {}
    #     P_sc = {}
    #     sampling_probs = np.zeros(num_molecules)
        
    #     for s in unique_clusters:
    #         # Indices of molecules in cluster s
    #         idx_s = np.where(cluster_ids == s)[0]
    #         labels_s = labels[idx_s]
    #         classes_in_s = np.unique(labels_s)
            
    #         # Compute N_{s,c}, w_{s,c}, and sum_w_c for normalization
    #         N_c = {}
    #         w_c = {}
    #         sum_w_c = 0.0
    #         for c in classes_in_s:
    #             N_c[c] = np.sum(labels_s == c)
    #             N_sc[(s, c)] = N_c[c]
    #             w_c[c] = 1.0 / (N_c[c] + epsilon)
    #             sum_w_c += w_c[c]
                
    #         # Normalize w_c to get P_{s,c} (sampling probability for classes within cluster s)
    #         for c in classes_in_s:
    #             P_sc[(s, c)] = w_c[c] / sum_w_c
            
    #         # Step 4: Assign sampling probability P_i to each molecule
    #         for idx in idx_s:
    #             c = labels[idx]
    #             P_i = (P_s[s]) * P_sc[(s, c)] / N_sc[(s, c)]
    #             sampling_probs[idx] = P_i
                
    #         if plot and plot_path is not None:
    #             dir_path = os.path.dirname(plot_path)
    #             os.makedirs(dir_path, exist_ok=True)

    #             # Prepare data for plotting
    #             # Create a dictionary to hold counts for each cluster and class
    #             cluster_class_counts = {}
    #             for s in unique_clusters:
    #                 cluster_class_counts[s] = {}
    #                 for c in unique_classes:
    #                     cluster_class_counts[s][c] = N_sc.get((s, c), 0)

    #             # Prepare data for stacked bar plot
    #             clusters = sorted(unique_clusters)
    #             classes = sorted(unique_classes)
    #             counts_per_class = {c: [] for c in classes}
    #             for s in clusters:
    #                 for c in classes:
    #                     counts_per_class[c].append(cluster_class_counts[s][c])

    #             # Plotting
    #             bar_width = 0.8
    #             fig, ax = plt.subplots(figsize=(12, 6))

    #             bottom = np.zeros(len(clusters))
    #             for c in classes:
    #                 ax.bar(clusters, counts_per_class[c], bottom=bottom, width=bar_width, label=f'Class {c}')
    #                 bottom += counts_per_class[c]

    #             ax.set_xlabel('Scaffold Cluster ID')
    #             ax.set_ylabel('Count')
    #             ax.set_title('Class Distribution in Each Scaffold Cluster')
    #             ax.legend(title='Class')
    #             # plt.xticks(clusters)
    #             ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #             # save the figure
    #             plt.savefig(plot_path)
    #             plt.close(fig)
                
    #     # Step 5: Normalize sampling probabilities so they sum to 1
    #     total_prob = np.sum(sampling_probs)
    #     sampling_probs /= total_prob
        
    #     print('Sum of all probabilities all molecule',sum(sampling_probs))
        
    #     sampled_indices = np.random.choice(num_molecules, size=N, replace=False, p=sampling_probs)
    #     sampled_data_indices = [train_idx[idx].item() for idx in sampled_indices]
        
    #     return sampled_data_indices

    def scaffold_clustering(self, cutoff):
        # Implement your scaffold clustering logic here
        # encoding scaffolds
        train_idx = self.split_dict['train']
        # print(train_idx)
        train_scaff_list = [self.scaff_list[i] for i in train_idx]
        
        scaff_mols = [Chem.MolFromSmiles(scaffold) for scaffold in train_scaff_list]
        
        ecfps = []
        # error = []
        for mol in scaff_mols:
            if mol is None:
                raise ValueError('Invalid Scaffold SMILES. Please check.')
            try:
                ecfps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            except Exception as e:
                raise ValueError(f'Error generating Morgan fingerprint: {e}')
        
        dists = calc_distance_matrix(ecfps)
        clusters = Butina.ClusterData(dists, len(ecfps), cutoff, isDistData=True)
        self.cluster_ids = [0] * len(ecfps)

        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                self.cluster_ids[idx] = cluster_id
        
        print(f'Clustered {len(ecfps)} training data into {len(clusters)} clusters.')
        
        return self.cluster_ids



def calc_distance_matrix(fps):
    """
    Calculate the upper triangle of the distance matrix for a list of fingerprints.

    Parameters:
    - fps: List of RDKit fingerprint objects.

    Returns:
    - dists: List of distances (1 - Tanimoto similarity) between fingerprints.
    """
    nfps = len(fps)
    dists = []
    for i in range(1, nfps):
        # Compute Tanimoto similarities between the i-th fingerprint and all previous fingerprints
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        # Convert similarities to distances and add to the list
        dists.extend([1 - x for x in sims])
    return dists