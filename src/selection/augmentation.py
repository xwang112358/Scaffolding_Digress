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
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add this line at the top of the file with other imports
__all__ = ['AugmentationDatasetSelector', 'SMILESRoundTripChecker', 'AugmentationDataset']


class AugmentationDatasetSelector:
    def __init__(self, name, root, smiles_list, y_list):
        self.name = name
        self.root = root
        self.smiles_list = smiles_list
        self.y_list = y_list
        self.scaff_list = [_generate_scaffold(smi) for smi in smiles_list]
        self.cluster_ids = None

    def scaffold_clustering(self, n_clusters=500):
        """
        Cluster scaffolds using MiniBatchKMeans 
        
        Parameters:
        - n_clusters: Number of clusters to generate
        
        Returns:
        - cluster_ids: List of cluster assignments for each molecule
        """
        print('extracting scaffolds')
        scaff_mols = [Chem.MolFromSmiles(scaffold) for scaffold in self.scaff_list]
        fps = []
        for mol in scaff_mols:
            if mol is None:
                raise ValueError('Invalid Scaffold SMILES. Please check.')
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                # Convert fingerprint to numpy array
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            except Exception as e:
                raise ValueError(f'Error generating Morgan fingerprint: {e}')
        
        # Store fingerprints for visualization
        self.fps = np.array(fps)
        
        print('clustering fingerprints')
        # Initialize and fit MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(3 * n_clusters, len(self.fps)),
            n_init='auto'
        )
        self.cluster_ids = kmeans.fit_predict(self.fps)
        
        print(f'Clustered {len(fps)} data into {n_clusters} clusters.')

        cluster_sizes = np.bincount(self.cluster_ids)
        print(f'largest cluster size: {np.max(cluster_sizes)}')
        print(f'smallest cluster size: {np.min(cluster_sizes)}')
        print(f'average cluster size: {np.mean(cluster_sizes)}')
        
        return self.cluster_ids
    

    def active_sampling(self, N, seed=42):
        """
        Uniformly sample active molecules with replacement.

        Parameters:
        - N: Number of samples to draw.
        - seed: Random seed for reproducibility.

        Returns:
        - selected_data: DataFrame containing sampled active molecules.
        """
        np.random.seed(seed)
        
        # Filter indices of active molecules
        active_indices = [i for i, label in enumerate(self.y_list) if label == 1]
        
        if not active_indices:
            raise ValueError("No active molecules found with the specified label.")
        
        # Uniformly sample active molecules with replacement
        sampled_indices = np.random.choice(active_indices, size=N, replace=True)
        
        selected_smiles = [self.smiles_list[idx] for idx in sampled_indices]
        selected_labels = [self.y_list[idx] for idx in sampled_indices]
        selected_scaffolds = [self.scaff_list[idx] for idx in sampled_indices]
        selected_data = pd.DataFrame({'smiles': selected_smiles, 'scaffold': selected_scaffolds, 'y': selected_labels})

        return selected_data

    
    def SABS_sampling(self, N, seed=42):
        """
        Scaffold-aware Balanced Sampling(SABS)
        """
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

        return selected_data
    
    def plot_clustering(self, save_path=None, method='pca'):
        """
        Visualize clustering results using dimensionality reduction
        
        Parameters:
        - save_path: Path to save the plot. If None, displays the plot instead.
        - method: 'tsne' or 'pca' for visualization method
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import seaborn as sns
        
        if self.cluster_ids is None:
            raise ValueError("Must run scaffold_clustering before plotting")
        
        if not hasattr(self, 'fps'):
            raise ValueError("Fingerprints not found. Must run scaffold_clustering first")
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            print('Running t-SNE...')
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(self.fps)-1),
                n_iter=1000,
                learning_rate='auto',
                init='pca'
            )
            method_name = 't-SNE'
        else:  # PCA
            print('Running PCA...')
            reducer = PCA(n_components=2, random_state=42)
            method_name = 'PCA'
        
        X_2d = reducer.fit_transform(self.fps)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                             c=self.cluster_ids, 
                             cmap='tab20',
                             alpha=0.6)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'{method_name} visualization of scaffold clusters')
        plt.xlabel(f'{method_name} dimension 1')
        plt.ylabel(f'{method_name} dimension 2')
        
        # Add legend showing number of clusters
        n_clusters = len(np.unique(self.cluster_ids))
        info_text = f'Number of clusters: {n_clusters}\n'
        info_text += f'Total points: {len(self.fps)}'
        
        if method.lower() == 'pca':
            # Add explained variance ratio for PCA
            var_ratio = reducer.explained_variance_ratio_
            info_text += f'\nExplained variance:\n'
            info_text += f'PC1: {var_ratio[0]:.3f}\n'
            info_text += f'PC2: {var_ratio[1]:.3f}'
        
        plt.text(0.02, 0.98, info_text,
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ----------------- AugmentationDataset -----------------
atom_decoder = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']


class AugmentationDataset(InMemoryDataset):
    """
    Construct an augmentation dataset from a list of smiles and labels 
    sampled by SABS Algorithm
    """
    def __init__(self, cfg, 
                 smiles_list, 
                 label_list, 
                 filter_dataset = False, 
                 transform=None, 
                 pre_transform=None):
        
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
        """
        Convert the list of smiles and labels into a PyTorch Geometric Data object
        """
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        data_list = []
        smiles_kept = []
        # smiles2graph
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
                # skip the molecule if it contains atom not in atom_decoder
                continue
            
            # scaffold node mask
            node_mask = torch.zeros(N, dtype=torch.bool)
            for idx in scaffold_indices:
                node_mask[idx] = True
            
            row, col, edge_type = [], [], []
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

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.tensor([self.label_list[i]], dtype=torch.int)
            self.new_label_list.append(self.label_list[i])
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, 
                        node_mask=node_mask)
            
            smiles_kept.append(smile)
            data_list.append(data)
            # save the kept smiles
        torch.save(smiles_kept, os.path.join(self.root, self.name, f'{self.name}_{self.split_scheme}_{self.ratio}_selected_filtered_smiles.pt'))
        torch.save(self.collate(data_list), self.processed_paths[0])
    

    def get_label_list(self):
        return self.new_label_list


                        
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