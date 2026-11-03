"""Original QIKT CLI with post-training test inference disabled locally.

Model selection and early stopping are unchanged. Frozen candidate test
evaluation remains a separate step using the repository prediction entrypoint.
"""
from pathlib import Path
import runpy
import sys
import os


def initialize_from_baseline(model, checkpoint):
    import torch
    if not model.model.freeze_backbone:
        raise ValueError('baseline warm-start is scoped to frozen refinements')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=False)
    allowed=('model.item_residual.','model.concept_memory_residual.','model.head_fusion_residual.')
    if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
        raise ValueError(f'Unexpected warm-start mismatch: {result}')
    return model


def main():
    examples = Path(__file__).resolve().parents[2] / 'examples'
    sys.path.insert(0, str(examples))
    import wandb_train
    wandb_train.predict_after_train = 0
    warm_start=os.environ.get('QIKT_SWITCH_BASE_CHECKPOINT')
    if warm_start:
        original_model_init=wandb_train.init_model
        def init_warm(*args,**kwargs):
            return initialize_from_baseline(original_model_init(*args,**kwargs),warm_start)
        wandb_train.init_model=init_warm
    overlap_path = os.environ.get('QIKT_SWITCH_OVERLAP_TRAIN')
    if overlap_path:
        from pykt.datasets.que_data_loader import KTQueDataset
        from torch.utils.data import DataLoader
        original_init = wandb_train.init_dataset4train

        def init_overlap(dataset_name, model_name, data_config, i, batch_size, **kwargs):
            if model_name != 'qikt_mamba':
                raise ValueError('overlap override is scoped to qikt_mamba')
            loaders = original_init(dataset_name, model_name, data_config, i, batch_size, **kwargs)
            cfg = data_config[dataset_name]
            train = KTQueDataset(overlap_path, input_type=cfg['input_type'],
                folds=set(cfg['folds']) - {i}, concept_num=cfg['num_c'],
                max_concepts=cfg['max_concepts'], mode='train', aug_probs=kwargs.get('aug_probs'))
            return (DataLoader(train, batch_size=batch_size), *loaders[1:])

        wandb_train.init_dataset4train = init_overlap
    runpy.run_path(str(examples / 'wandb_qikt_mamba_train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
