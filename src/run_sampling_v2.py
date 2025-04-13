import os
import time
from welqrate.dataset import WelQrateDataset
from selection.augmentation import AugmentationDatasetSelector_v2
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
    sampling_method = cfg.augment_data.sampling_method
    print(f'name: {name} \n root: {root} \n split_scheme: {split_scheme} \n ratio: {ratio}')

    # load train data 
    welqrate_dataset = WelQrateDataset(dataset_name=name, root=root, mol_repr='2dmol')
    split_dict = welqrate_dataset.get_idx_split(split_scheme=split_scheme)
    train_data = welqrate_dataset[split_dict['train']]
    num_train_samples = len(train_data)
    # only use active molecules for augmentation
    train_active_data = [data for data in train_data if data.y != 0]
    num_train_active_samples = len(train_active_data)
    train_active_smiles = [data.smiles for data in train_active_data]
    print(f'number of active samples: {num_train_active_samples}')

    os.makedirs(f'./sampled_smiles/{sampling_method}', exist_ok=True)
    save_path = f'sampled_smiles_{name}_{split_scheme}_{ratio}.csv'
    if os.path.exists(f'./sampled_smiles/{sampling_method}/{save_path}'):
        print('Loading saved sampled smiles')
        sampled_smiles_df = pd.read_csv(f'./sampled_smiles/{sampling_method}/{save_path}')
        print('smiles has been sampled for this setting')
    else:
        print('start constructing scaffold library')
        augment_selector = AugmentationDatasetSelector_v2(name=name, root=root, smiles_list=train_active_smiles)
        print('start scaffold clustering')
        start_time = time.time()
        N = int(num_train_samples * ratio)
        # scaffold-aware balanced sampling
        if cfg.augment_data.sampling_method == 'sabs':
            cluster_ids, optimal_n_clusters = augment_selector.scaffold_clustering(min_clusters=10, max_clusters=30)  
            print('scaffold clustering time:', time.time() - start_time)
            
            print('start sampling')
            start_time = time.time()
            sampled_smiles_df = augment_selector.sabs_sampling(N=N, seed=42)
            print('sampling time:', time.time() - start_time)
        elif cfg.augment_data.sampling_method == 'uniform_active':
            sampled_smiles_df = augment_selector.active_sampling(N=N, seed=42)
            print('sampling time:', time.time() - start_time)
        else:
            raise ValueError(f"Invalid sampling method: {cfg.augment_data.sampling_method}")
        
        print(f'Saving sampled molecules and related scaffolds to {save_path}')
        sampled_smiles_df.to_csv(f'./sampled_smiles/{sampling_method}/{save_path}', index=False)
    
if __name__ == '__main__':
    main()