# import graph_tool as gt
import os
import pathlib
import warnings

import torch
from torch_geometric.loader import DataLoader
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete_aug import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from selection.augmentation import AugmentationDatasetSelector, AugmentationDataset
from selection.get_tdc_dataset import get_tdc_dataset

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

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        elif dataset_config.name == 'welqrate':
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
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    elif cfg.model.type == 'discrete' and cfg.general.setting == 'augment':
        print(111111111111111111111111111)
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
    
    # load the dataset for augmentation
    name = cfg.augment_data.name
    root = cfg.augment_data.data_dir
    # print current working directory
    # print("Current working directory: {0}".format(os.getcwd()))
    
    data_dict = get_tdc_dataset(name, root)
    train_data = data_dict['train']
    train_smiles = [data['smiles'] for data in train_data]
    train_y = [data['y'].item() for data in train_data]
    augment_selector = AugmentationDatasetSelector(name = name , root = root, smiles_list = train_smiles, y_list = train_y)
    cluster_ids = augment_selector.scaffold_clustering(cutoff=0.4)
    sampled_smiles_df = augment_selector.augmentation_sampling(N = 100, seed = 42)
    sampled_smiles = sampled_smiles_df['smiles'].tolist()
    selected_data = AugmentationDataset(name = name, root = root, smiles_list = sampled_smiles,)

    augment_loader = DataLoader(selected_data, batch_size=cfg.augment_data.batch_size, 
                                shuffle=False)
    
    # for i in range(5):
    #     print(selected_data[i])
        
    # for batch in augment_loader:
    #     print(batch)
    #     break

    if cfg.general.setting == 'train_scratch':
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume) # resume: null

    elif cfg.general.setting == 'train_continue':
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)

    # replace the datamodule with dataloader = 

    elif cfg.general.setting in ['test', 'augment']:
        trainer.test(model, dataloaders=augment_loader, ckpt_path=cfg.general.ckpt_path)



if __name__ == '__main__':
    main()
