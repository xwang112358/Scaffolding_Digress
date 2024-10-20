from .polymer import PolymerRegDataset
from ogb.graphproppred import PygGraphPropPredDataset
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch
from .scaffold import _generate_scaffold, generate_scaffolds_dict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_dataset(args, load_path, load_unlabeled_name="None"):
    if load_unlabeled_name=='None':
        if args.dataset.startswith('plym'):
            return PolymerRegDataset(args.dataset, load_path)
        elif args.dataset.startswith('ogbg'):
            ogbg_dataset = PygGraphPropPredDataset(args.dataset, load_path)
            label_split_idx = ogbg_dataset.get_idx_split()
            meta_info = ogbg_dataset.meta_info
            ogbg_data_list = [data for data in ogbg_dataset]
            smile_path = os.path.join('./raw_data', '_'.join(args.dataset.split('-')), 'mapping/mol.csv.gz')
            smiles = pd.read_csv(smile_path, compression='gzip', usecols=['smiles'])
            smiles_list = smiles['smiles'].tolist()
            
            new_dataset = augmentation_dataset(args.dataset, load_path, ogbg_data_list, smiles_list, label_split_idx, meta_info)
            
            return new_dataset #PygGraphPropPredDataset(args.dataset, load_path)
    else:
        raise ValueError('Unlabeled dataset {} not supported'.format(load_unlabeled_name))
    
    
class augmentation_dataset(InMemoryDataset):
    def __init__(self, name, root, data_list, smile_list, split_dict, meta_dict=None,
                 transform=None, pre_transform=None):
        self.name = '_'.join(name.split('-'))
        self.root = root
        self.smile_list = smile_list
        self.total_data_len = len(data_list)
        self.data_list = data_list
        self.meta_info = meta_dict
        self.split_dict = split_dict
        self.scaff_list = [_generate_scaffold(smi) for smi in smile_list]
        

        super(augmentation_dataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Return an empty list since raw files are not used
        return []

    @property
    def processed_file_names(self):
        # Define the name of the processed file
        return [f'{self.name}_scaff_processed.pt']

    @property
    def processed_dir(self):
        # Override to save processed files directly in the root directory
        return os.path.join(self.root, self.name, 'processed')
    


    def process(self):
        # Implement your processing logic here
        # _, self.scaffold_sets = generate_scaffolds_dict(self.smile_list)
        print(self.data_list[0])
        # self.scaff_list = []
        for i in tqdm(range(len(self.data_list))):
            self.data_list[i].smiles = self.smile_list[i]
            # scaff_smiles = _generate_scaffold(self.smile_list[i])
            # self.scaff_list.append(scaff_smiles)
            self.data_list[i].scaff_smiles = self.scaff_list[i]

        print(self.data_list[:10])
        print(len(self.data_list))
        data, slices = self.collate(self.data_list)
        # print(data)
        # print(slices)
        # Save the processed data to the root directory
        torch.save((data, slices), self.processed_paths[0])
        
        # self.data, self.slices = torch.load(self.processed_paths[0])


    def augmentation_sampling(self, N, seed=42, plot=True, plot_path=None):
        # return the indices of the sampled, training data
        
        # calculate the joint distribution of the scaffold and the label for training data
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        train_idx = self.split_dict['train']
        num_molecules = len(self.cluster_ids)
        
        # Get cluster IDs and labels for training molecules
        cluster_ids = np.array(self.cluster_ids)
        print(cluster_ids)
        labels = np.array([self.data_list[i].y.item() for i in train_idx])
        
        unique_clusters = np.unique(cluster_ids)
        unique_classes = np.unique(labels)
        print(f'Unique clusters: {len(unique_clusters)}, Unique classes: {len(unique_classes)}')
        
        # Step 1: Compute N_s (number of molecules in each scaffold cluster)
        N_s = {}
        for s in unique_clusters:
            N_s[s] = np.sum(cluster_ids == s)
        
        # Step 2: Compute w_s (inverse frequency weights for scaffold clusters)
        epsilon = 1e-6
        w_s = {}
        for s in unique_clusters:
            w_s[s] = 1.0 / (N_s[s] + epsilon)
        
        # Normalize w_s to get P_s (sampling probability for scaffold clusters)
        sum_w_s = sum(w_s.values())
        P_s = {s: w_s[s] / sum_w_s for s in unique_clusters}
        
        # Step 3: Compute N_{s,c} (number of molecules for each class within each scaffold cluster)
        N_sc = {}
        w_sc = {}
        P_sc = {}
        sampling_probs = np.zeros(num_molecules)
        
        for s in unique_clusters:
            # Indices of molecules in cluster s
            idx_s = np.where(cluster_ids == s)[0]
            labels_s = labels[idx_s]
            classes_in_s = np.unique(labels_s)
            
            # Compute N_{s,c}, w_{s,c}, and sum_w_c for normalization
            N_c = {}
            w_c = {}
            sum_w_c = 0.0
            for c in classes_in_s:
                N_c[c] = np.sum(labels_s == c)
                N_sc[(s, c)] = N_c[c]
                w_c[c] = 1.0 / (N_c[c] + epsilon)
                sum_w_c += w_c[c]
                
            # Normalize w_c to get P_{s,c} (sampling probability for classes within cluster s)
            for c in classes_in_s:
                P_sc[(s, c)] = w_c[c] / sum_w_c
            
            # Step 4: Assign sampling probability P_i to each molecule
            for idx in idx_s:
                c = labels[idx]
                P_i = (P_s[s]) * P_sc[(s, c)] / N_sc[(s, c)]
                sampling_probs[idx] = P_i
                
            if plot and plot_path is not None:
                os.makedirs(plot_path, exist_ok=True)

                # Prepare data for plotting
                # Create a dictionary to hold counts for each cluster and class
                cluster_class_counts = {}
                for s in unique_clusters:
                    cluster_class_counts[s] = {}
                    for c in unique_classes:
                        cluster_class_counts[s][c] = N_sc.get((s, c), 0)

                # Prepare data for stacked bar plot
                clusters = sorted(unique_clusters)
                classes = sorted(unique_classes)
                counts_per_class = {c: [] for c in classes}
                for s in clusters:
                    for c in classes:
                        counts_per_class[c].append(cluster_class_counts[s][c])

                # Plotting
                bar_width = 0.8
                fig, ax = plt.subplots(figsize=(12, 6))

                bottom = np.zeros(len(clusters))
                for c in classes:
                    ax.bar(clusters, counts_per_class[c], bottom=bottom, width=bar_width, label=f'Class {c}')
                    bottom += counts_per_class[c]

                ax.set_xlabel('Scaffold Cluster ID')
                ax.set_ylabel('Count')
                ax.set_title('Class Distribution in Each Scaffold Cluster')
                ax.legend(title='Class')
                # plt.xticks(clusters)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # save the figure
                plt.savefig(os.path.join(plot_path, 'class_distribution_per_cluster.png'))
                plt.close(fig)
                
        # Step 5: Normalize sampling probabilities so they sum to 1
        total_prob = np.sum(sampling_probs)
        sampling_probs /= total_prob
        
        sampled_indices = np.random.choice(num_molecules, size=N, replace=True, p=sampling_probs)
        sampled_data_indices = [train_idx[idx].item() for idx in sampled_indices]
        
        return sampled_data_indices

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