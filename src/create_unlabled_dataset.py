
from selection.augmentation import AugmentationDatasetSelector, AugmentationDataset
from selection.get_tdc_dataset import get_tdc_dataset

name = 'HIA_Hou'
root = '../TDC_Dataset'

data_dict = get_tdc_dataset(name, root)

# print(data_list)
train_data = data_dict['train']
print(train_data[:5])
train_smiles = [data['smiles'] for data in train_data]
train_y = [data['y'].item() for data in train_data]
print(train_smiles[:5])
print(train_y[:5])

augment_selector = AugmentationDatasetSelector(name = name , root = root, smiles_list = train_smiles, y_list = train_y)


# visualize the clustering results to determine the cutoff value
cluster_ids = augment_selector.scaffold_clustering(cutoff=0.4)


sampled_smiles_df = augment_selector.augmentation_sampling(N = 100, seed = 42)
sampled_smiles = sampled_smiles_df['smiles'].tolist()

selected_data = AugmentationDataset(name = name, root = root, smiles_list = sampled_smiles,
                                    )

for i in range(5):
    print(selected_data[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
## 

import torch
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.utils import to_networkx

def visualize_scaffold_mask(data, smiles):
    """
    Visualize the molecular graph with scaffold highlighted based on node and edge masks.
    :param data: PyG Data object containing the graph and scaffold masks.
    :param smiles: SMILES string of the molecule.
    """
    # Convert PyG graph to a NetworkX graph for visualization
    G = to_networkx(data, to_undirected=True)

    # Get node positions using RDKit for better molecular layout
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    atom_pos = Draw.MolToImage(mol, size=(300, 300), kekulize=True)

    # Define node and edge colors based on masks
    node_colors = ["red" if data.node_mask[i] else "blue" for i in range(data.num_nodes)]
    edge_colors = ["red" if data.edge_mask[j] else "blue" for j in range(data.edge_index.size(1))]

    # Plot the molecular graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # You may replace with `atom_pos` if positions are extracted from RDKit.
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    plt.title(f"Molecule with SMILES: {smiles}\nRed = Scaffold, Blue = Non-scaffold")
    plt.axis("off")
    plt.show()


