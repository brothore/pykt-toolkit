"""AKT migration data/model interfaces; native labels and gap alignment."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def build_model(config, device='cuda'):
    kwargs = dict(config['model_config'])
    kwargs.pop('learning_rate', None)
    cfg = config['data_config']
    kwargs.update(n_question=cfg['num_c'], n_pid=cfg['num_q'], emb_type='qid')
    if config['mechanism'] == 'baseline':
        from pykt.models.akt import AKT
        model = AKT(**kwargs)
    else:
        from pykt.models.akt_fairness import AKTFairness
        model = AKTFairness(**kwargs, mechanism=config['mechanism'], scale=config['scale'])
    return model.to(device)


def loaders(config):
    from pykt.datasets.dkt_forget_dataloader import DktForgetDataset
    cfg = config['data_config']
    path = str(Path(cfg['dpath']) / cfg['train_valid_file'])
    # Identical concept-expanded native fragments in every arm, including baseline.
    train = DktForgetDataset(path, cfg['input_type'], set(cfg['folds']) - {config['params']['fold']})
    valid = DktForgetDataset(path, cfg['input_type'], {config['params']['fold']})
    return tuple(DataLoader(ds, batch_size=config['params']['batch_size'], num_workers=0)
                 for ds in (train, valid))


def predict(model, data, gaps=None):
    def full(key, mapping):
        return torch.cat((mapping[key][:, :1], mapping['shft_' + key]), 1).long()
    c, r, q = (full(k, data) for k in ['cseqs', 'rseqs', 'qseqs'])
    if type(model).__name__ == 'AKT':
        return model(c, r, q)
    full_gaps = {k: full(k, gaps) for k in ['rgaps', 'sgaps', 'pcounts']} if gaps else None
    return model(c, r, q, gaps=full_gaps)
