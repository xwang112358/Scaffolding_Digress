from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

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

def butina_clustering(fps, cutoff=0.2):
    """
    Perform Butina clustering on a list of fingerprints.

    Parameters:
    - fps: List of RDKit fingerprint objects.
    - cutoff: Distance threshold for clustering (float between 0.0 and 1.0).

    Returns:
    - cluster_ids: List of cluster IDs corresponding to each fingerprint.
    """
    # Calculate the distance matrix
    dists = calc_distance_matrix(fps)
    
    # Perform Butina clustering
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    
    # Initialize cluster ID list
    cluster_ids = [0] * len(fps)
    
    # Assign cluster IDs to each fingerprint
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_ids[idx] = cluster_id
            
    return cluster_ids

# Example usage:
# Assuming you have a list of SMILES strings and have generated fingerprints

# List of SMILES strings
smiles_list = ['CCO', 'CCN', 'CCC', 'CCCl', 'CCBr']  # Replace with your list of SMILES

# Generate ECFP fingerprints
fps = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fps.append(fp)

# Set the cutoff parameter (adjust based on your data)
cutoff = 0.5  # Commonly used values are between 0.2 and 0.7

# Perform clustering
cluster_ids = butina_clustering(fps, cutoff)

# Output cluster IDs
for idx, cluster_id in enumerate(cluster_ids):
    print(f'Molecule {idx} (SMILES: {smiles_list[idx]}) is in cluster {cluster_id}')
