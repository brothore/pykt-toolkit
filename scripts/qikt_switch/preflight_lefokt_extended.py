"""Check new source ablations with real native fragments before any retraining."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pykt.utils import set_seed
from reference_ablation_models import load_reference,intervene,predict
from reference_training import to_device
from prediction_progress import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    registration=json.loads((root/'registration.json').read_text())
    config=json.loads(Path(registration['original_config']).read_text());cfg=config['data_config'];fold=config['params']['fold']
    from pykt.datasets.data_loader import KTDataset
    dataset=KTDataset(registration['data_file'],cfg['input_type'],set(cfg['folds'])-{fold},mode='test',aug_probs={})
    batch,gaps=to_device(next(iter(DataLoader(dataset,batch_size=32))),'cuda');checks=[]
    for mode in registration['modes']:
        set_seed(config['params']['seed']);model,module=load_reference(registration['source'],config)
        patch=intervene(model,module,mode)
        model.eval()
        with torch.no_grad():score=predict(model,batch)
        if not torch.isfinite(score).all():raise ValueError('Nonfinite initial predictions')
        optimizer=torch.optim.Adam(model.parameters(),lr=float(config['model_config']['learning_rate']))
        model.train();optimizer.zero_grad()
        prediction=predict(model,batch)[:,1:];mask=batch['smasks'].bool()
        loss=torch.nn.functional.binary_cross_entropy(prediction[mask].double(),batch['shft_rseqs'][mask].double())
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite real-data gradient')
        optimizer.step();checks.append({'mode':mode,'finite_real_batch_and_gradient':True,'parameters':sum(p.numel() for p in model.parameters())})
        if patch:(root/(mode+'_preflight_forward.py')).write_text(patch)
        del model,optimizer
    if len({c['parameters'] for c in checks})!=1:raise ValueError('Trainable parameter counts unexpectedly differ')
    atomic_json(root/'native_preflight.json',{'passed':True,'checks':checks,'fragments':32,'scientific_training':False})


if __name__=='__main__':main()
