# import graph_tool as gt
import os
import pathlib
import warnings
import time
import torch
from torch_geometric.loader import DataLoader
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import pandas as pd

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from src.diffusion_model_discrete_scaffold import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from selection.augmentation import AugmentationDataset

warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='../configs', config_name='welqrate')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print(cfg)

    if dataset_config["name"] in ['qm9', 'guacamol', 'moses', 'welqrate']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config.name == 'welqrate':
            print("Loading welqrate dataset")
            from datasets import welqrate_dataset
            datamodule = welqrate_dataset.WelQrateDataModule(cfg)
            dataset_infos = welqrate_dataset.WelQrateinfos(datamodule, cfg)
            train_smiles = None    
        else:
            raise ValueError("Dataset not implemented")
            
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete' and cfg.general.setting != 'augment':
        # add the pretraining guidance 
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    elif cfg.model.type == 'discrete' and cfg.general.setting == 'augment':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs, augment=True, 
                                           max_aug_steps=cfg.augment_data.max_aug_steps)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    use_gpu = 1 > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=True,
                      callbacks=callbacks,
                      log_every_n_steps=50,
                      logger = [])
    
    # load the selected dataset for augmentation
    name = cfg.augment_data.name
    split_scheme = cfg.augment_data.split
    ratio = cfg.augment_data.ratio
    print(ratio)

    ### ------------- Augmentation ------------- ###
    sampled_smiles_path = f'sampled_smiles_{name}_{split_scheme}_{ratio}.csv'
    sampled_smiles_df = pd.read_csv(sampled_smiles_path)
    sampled_scaffolds = sampled_smiles_df['scaffold'].tolist()
    # sampled_smiles = sampled_smiles_df['smiles'].tolist()
    sampled_labels = sampled_smiles_df['y'].tolist()  
    
    selected_data = AugmentationDataset(cfg = cfg, smiles_list = sampled_scaffolds, 
                                        label_list = sampled_labels)

    augment_loader = DataLoader(selected_data, batch_size=cfg.augment_data.batch_size, 
                                shuffle=False)
    
    sampled_labels = selected_data.get_label_list()
    print(sampled_labels)

    if cfg.general.setting == 'train_scratch':
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume) # resume: null

    elif cfg.general.setting == 'train_continue':
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)

    elif cfg.general.setting in ['test', 'augment']:
        trainer.test(model, dataloaders=augment_loader, ckpt_path=cfg.general.ckpt_path)
        generated_graphs = model.get_augmented_graphs()

        # assign formal charge and explicit Hydrogen to the generated graphs

        # check the validity of the generated graphs
        from analysis.rdkit_functions import MoleculeValidator
        metrics = MoleculeValidator(atom_decoder=dataset_infos.atom_decoder)
        metrics_dict = metrics.process_molecules(generated_graphs)
        print('validity', metrics_dict['validity'])
        print('relaxed_validity', metrics_dict['relaxed_validity'])

        # for i in range(len(generated_graphs)): list index out of range
        #     generated_graphs[i].y = torch.tensor(sampled_labels[i], dtype=torch.int)
        torch.save(generated_graphs, f'{name}_{split_scheme}_{ratio}_generated_graphs.pt')        
        print(generated_graphs[0])

    
    



if __name__ == '__main__':
    main()
