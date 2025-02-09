import os
import time
from welqrate.dataset import WelQrateDataset
from selection.augmentation import AugmentationDatasetSelector
import hydra
from omegaconf import DictConfig
import pandas as pd
        
@hydra.main(version_base='1.3', config_path='../configs', config_name='welqrate')
def main(cfg: DictConfig):

    """Get augmentation data based on configuration settings."""
    name = cfg.augment_data.name
    root = cfg.augment_data.data_dir
    split_scheme = cfg.augment_data.split
    ratio = cfg.augment_data.ratio
    print(f'name: {name} \n root: {root} \n split_scheme: {split_scheme} \n ratio: {ratio}')

    welqrate_dataset = WelQrateDataset(dataset_name=name, root=root, mol_repr='2dmol')
    split_dict = welqrate_dataset.get_idx_split(split_scheme=split_scheme)
    train_data = welqrate_dataset[split_dict['train']]
    train_smiles = train_data.smiles
    num_train_samples = len(train_smiles)
    train_y = train_data.y.tolist()

    os.makedirs('./sampled_smiles', exist_ok=True)
    save_path = f'sampled_smiles_{name}_{split_scheme}_{ratio}.csv'
    if os.path.exists(f'./sampled_smiles/{save_path}'):
        print('Loading saved sampled smiles')
        sampled_smiles_df = pd.read_csv(f'./sampled_smiles/{save_path}')
        print('smiles has been sampled for this setting')
    else:
        print('start selecting reference smiles')
        augment_selector = AugmentationDatasetSelector(name=name, root=root, smiles_list=train_smiles, y_list=train_y)
        print('start scaffold clustering')
        start_time = time.time()
        cluster_ids = augment_selector.scaffold_clustering(cutoff=0.4)  # maximum cutoff for clustering
        print('scaffold clustering time:', time.time() - start_time)
        print('start sampling')
        start_time = time.time()
        N = int(num_train_samples * ratio)
        sampled_smiles_df = augment_selector.augmentation_sampling(N=N, seed=42)
        print('sampling time:', time.time() - start_time)
        
        sampled_smiles = sampled_smiles_df['smiles'].tolist()
        print('Saving sampled molecules and related scaffolds')
        
        sampled_smiles_df.to_csv(f'./sampled_smiles/{save_path}', index=False)
    

if __name__ == '__main__':
    main()