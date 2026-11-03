import copy
import pytest
import torch
import torch.nn.functional as F
from pykt.models.dkt import DKT
from pykt.models.dkt_joint import DKTJoint,COMPONENTS,TERMS
from pykt.models.train_model import model_forward


def configs(dataset):
    if dataset=='assist2009':return {'action':{'action_mode':'conditional','action_balance':.5,'action_class_counts':[4,2,1,1]},
          'process':{'process_mode':'joint','process_stats':{'time_center':1.,'time_scale':2.}},
          'distillation':{'kd_temperature':1.,'kd_mode':'uniform'},'transfer':{'transfer_loss':'bce'}}
    return {'confidence':{'confidence_mode':'ordinal','confidence_counts':[1,2,1,3,1]},
            'option':{'num_q':3,'option_mode':'direct','option_keys':[0,1,2]},'profile':{}}


def batch():
    m=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    return {'cseqs':torch.tensor([[0,1,0],[1,0,0]]),'rseqs':torch.tensor([[1.,0.,1.],[0.,1.,0.]]),
      'shft_cseqs':torch.tensor([[1,0,1],[0,1,0]]),'shft_rseqs':torch.tensor([[0.,1.,0.],[1.,0.,0.]]),'masks':m.clone(),'smasks':m.clone(),
      'shft_qseqs':torch.tensor([[0,1,2],[1,0,0]]),'action_labels':torch.tensor([[1,0,2],[0,3,-100]]),'action_mask':m.clone(),
      'process_time':torch.ones(2,3)*2,'process_attempt':torch.tensor([[1,2,3],[0,1,0]]),'process_hint':torch.tensor([[0.,1.,1.],[0.,0.,0.]]),
      'process_time_mask':m.clone(),'process_attempt_mask':m.clone(),'process_hint_mask':m.clone(),
      'teacher_probs':torch.tensor([[.2,.8,.4],[.7,.3,float('nan')]]),'teacher_mask':m.clone(),
      'future_concepts':torch.tensor([[[0,1],[1,-1],[0,-1]],[[1,-1],[0,1],[-1,-1]]]),
      'future_labels':torch.tensor([[1.,0.,1.],[0.,1.,float('nan')]]),'future_mask':m.clone(),
      'confidence_labels':torch.tensor([[0,4,2],[1,2,-100]]),'confidence_mask':m.clone(),
      'option_labels':torch.tensor([[1,1,0],[1,2,-100]]),'option_mask':m.clone(),
      'profile_anchors':torch.tensor([[0,0],[1,1]]),'profile_concepts':torch.tensor([[[.5,.5],[1.,0.]],[[0.,1.],[.25,.75]]]),
      'profile_labels':torch.tensor([[.5,1.],[0.,.25]]),'profile_mass':torch.tensor([[.5,.5],[.3,.7]]),'profile_mask':torch.ones(2,2,dtype=torch.bool)}


@pytest.mark.parametrize('dataset',['assist2009','nips_task34'])
def test_zero_aux_rng_main_bce_adam_export_and_single_backbone(dataset):
    data=batch();torch.manual_seed(55);old=DKT(2,8,.2);rng=torch.random.get_rng_state()
    torch.manual_seed(55);model=DKTJoint(2,8,.2,dataset_name=dataset,task_configs=configs(dataset),loss_weights={})
    assert torch.equal(rng,torch.random.get_rng_state());assert set(model.export_dkt_state_dict())==set(old.state_dict())
    assert len([name for name,_ in model.named_modules() if isinstance(_,torch.nn.LSTM)])==1
    a,b=torch.optim.Adam(old.parameters()),torch.optim.Adam(model.parameters());rng=torch.random.get_rng_state()
    p=old(data['cseqs'],data['rseqs'].long()).gather(-1,data['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    mask=data['smasks'];loss=F.binary_cross_entropy(p[mask].double(),data['shft_rseqs'][mask].double());loss.backward();a.step()
    torch.random.set_rng_state(rng);other=model.compute_loss({k:v for k,v in data.items() if k in ['cseqs','rseqs','shft_cseqs','shft_rseqs','masks','smasks']})
    assert torch.equal(loss,other);other.backward();b.step()
    for name,value in old.state_dict().items():assert torch.equal(value,model.export_dkt_state_dict()[name])


@pytest.mark.parametrize('task',list(COMPONENTS))
def test_each_original_component_is_exactly_preserved(task):
    data=batch();dataset='assist2009' if task in ('action','process','distillation','transfer') else 'nips_task34';cfg=configs(dataset)
    weights={t:.17 for t in TERMS[task]}
    torch.manual_seed(48);model=DKTJoint(2,8,.2,dataset_name=dataset,task_configs=cfg,loss_weights=weights)
    single_cfg=copy.deepcopy(cfg[task]);attrs={'action':['action_weight'],'process':['time_weight','attempt_weight','hint_weight'],
        'distillation':['kd_weight'],'transfer':['transfer_weight'],'confidence':['confidence_weight'],'option':['option_weight'],'profile':['profile_weight']}
    for attr in attrs[task]:single_cfg[attr]=.17
    torch.manual_seed(48);single=COMPONENTS[task](2,8,dropout=.2,**single_cfg)
    # Same head initialization, dropout mask and weighted loss, not a mirrored formula.
    rng=torch.random.get_rng_state();actual=model.compute_loss(data);torch.random.set_rng_state(rng);expected=single.compute_loss(data)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12),(task,actual,expected)
    actual.backward();expected.backward()
    for name,param in single.named_parameters():
        other=dict(model.named_parameters()).get(name) if name.startswith(('interaction_emb.','lstm_layer.','out_layer.')) else dict(model.named_parameters()).get('aux_heads.'+task+'.'+name)
        assert torch.equal(other.grad,param.grad),(task,name)


@pytest.mark.parametrize('dataset',['assist2009','nips_task34'])
def test_joint_single_forward_weighted_sum_gradients_and_no_inference_leakage(dataset,tmp_path):
    data=batch();cfg=configs(dataset);weights={t:.1 for task in cfg for t in TERMS[task]}
    model=DKTJoint(2,8,0,dataset_name=dataset,task_configs=cfg,loss_weights=weights)
    calls=[];hook=model.lstm_layer.register_forward_hook(lambda *args:calls.append(1))
    total,parts=model.compute_loss(data,return_components=True);hook.remove();assert len(calls)==1
    assert torch.equal(total,parts['response']+sum(weights[t]*parts[t] for t in weights))
    assert torch.equal(total,model_forward(model,data));total.backward()
    assert model.lstm_layer.weight_ih_l0.grad.abs().sum()>0
    for task,heads in model.aux_heads.items():
        for p in heads.parameters():assert p.grad is not None
    before=model(data['cseqs'],data['rseqs'].long());changed=copy.deepcopy(data)
    for key in ('action_labels','process_time','teacher_probs','future_labels','confidence_labels','option_labels','profile_labels'):changed[key].zero_()
    assert torch.equal(before,model(changed['cseqs'],changed['rseqs'].long()))
    old=DKT(2,8,0);old.load_state_dict(model.export_dkt_state_dict(),strict=True);assert torch.equal(before,old(data['cseqs'],data['rseqs'].long()))
    path=tmp_path/'state';torch.save(model.state_dict(),path);clone=DKTJoint(2,8,0,dataset_name=dataset,task_configs=cfg,loss_weights=weights)
    clone.load_state_dict(torch.load(path,weights_only=True),strict=True);assert torch.equal(total,clone.compute_loss(data))
    per=model.compute_loss(data,return_per_sample=True);assert per.shape==(2,) and torch.isfinite(per).all()


@pytest.mark.parametrize('weight',[-1,float('nan'),float('inf')])
def test_invalid_joint_weights(weight):
    with pytest.raises(ValueError):DKTJoint(2,8,dataset_name='nips_task34',task_configs=configs('nips_task34'),loss_weights={'option':weight})


def test_nips_joint_dataset_training_statistics_and_variable_groups(tmp_path):
    import pandas as pd
    from test_dkt_profile import fixture
    from pykt.datasets.joint_dataset import JointDataset,joint_collate
    fixture(tmp_path);rawpath=tmp_path/'train_task_3_4.csv';raw=pd.read_csv(rawpath)
    raw['CorrectAnswer']=raw.QuestionId.map({100:1,101:2,102:3})
    raw['AnswerValue']=raw.CorrectAnswer.where(raw.IsCorrect==1,raw.CorrectAnswer%4+1);raw.to_csv(rawpath,index=False)
    path=tmp_path/'train_valid_sequences.csv';train=JointDataset(path,['questions','concepts'],{1},'nips_task34')
    valid=JointDataset(path,['questions','concepts'],{0},'nips_task34',mode='valid',stats=train.stats)
    assert valid.stats==train.stats;assert train.profile.stats['sessions']==1 and valid.profile.stats['sessions']==0
    data=joint_collate([train[0],valid[0]])
    for key in train.base[0]:assert torch.equal(data[key][0],torch.as_tensor(train.base[0][key]))
    assert data['profile_mask'][1].sum()==0
    assert torch.equal(train.confidence.event_ids,train.option.event_ids)
    with pytest.raises(ValueError):JointDataset(path,['questions','concepts'],{0},'nips_task34',mode='valid')
    model=DKTJoint(2,8,0,dataset_name='nips_task34',task_configs=configs('nips_task34'),loss_weights={'option':.1,'confidence':.1,'profile':.03})
    assert torch.isfinite(model.compute_loss(data))


def test_assist_joint_dataset_teacher_boundary_and_alignment(tmp_path):
    import json,numpy as np,pandas as pd
    from test_dkt_process import _process_fixture
    from pykt.datasets.distill_dataset import build_history_sidecar
    from pykt.datasets.joint_dataset import JointDataset,joint_collate
    rawpath=_process_fixture(tmp_path);raw=pd.read_csv(rawpath);raw['template_id']=1;raw.to_csv(rawpath,index=False)
    history=build_history_sidecar(tmp_path)
    with np.load(history) as f:values={k:f[k].copy() for k in f.files}
    shape=values['events'][:,1:].shape;mask=np.ones(shape,dtype=bool);mask[values['folds']==0]=False
    teacher=tmp_path/'soft_targets.npz';np.savez(teacher,rich_probs=np.full(shape,.6,dtype=np.float32),thin_probs=np.full(shape,.6,dtype=np.float32),mask=mask,folds=values['folds'],uids=values['uids'])
    teacher.with_suffix('.json').write_text(json.dumps({'sources':json.loads(history.with_suffix('.json').read_text())['sources']}))
    path=tmp_path/'train_valid_sequences.csv';train=JointDataset(path,['questions','concepts'],{1},'assist2009',teacher_path=teacher)
    valid=JointDataset(path,['questions','concepts'],{0},'assist2009',mode='valid',stats=train.stats,teacher_path=teacher)
    assert train.distill.mask.any();assert not valid.distill.mask.any();assert valid.stats==train.stats
    assert torch.equal(train.first_action.event_ids,train.process.event_ids)
    data=joint_collate([train[0],train[1]])
    for key in train.base[0]:assert torch.equal(data[key][0],torch.as_tensor(train.base[0][key]))
    weights={t:.1 for task in configs('assist2009') for t in TERMS[task]}
    assert torch.isfinite(DKTJoint(2,8,0,dataset_name='assist2009',task_configs=configs('assist2009'),loss_weights=weights).compute_loss(data))


def test_joint_rejects_shifted_component_base():
    from pykt.datasets.joint_dataset import _check_shared_base
    from types import SimpleNamespace
    base={'cseqs':torch.tensor([[0,1]]),'rseqs':torch.tensor([[1.,0.]]),'uid':torch.tensor([1]),'masks':torch.tensor([[True]]),'smasks':torch.tensor([[True]])}
    bad=copy.deepcopy(base);bad['cseqs'][0,0]=1
    class Component:
        dori=bad
        def __len__(self):return 1
    with pytest.raises(AssertionError):_check_shared_base(base,Component(),'deliberately_shifted')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
@pytest.mark.parametrize('dataset',['assist2009','nips_task34'])
def test_eval_cuda_device_move_with_cpu_labels_and_buffers(dataset):
    data=batch();weights={t:.1 for task in configs(dataset) for t in TERMS[task]}
    cpu=DKTJoint(2,8,.2,dataset_name=dataset,task_configs=configs(dataset),loss_weights=weights).eval()
    gpu=copy.deepcopy(cpu).cuda().eval()
    assert all(buffer.device.type=='cuda' for buffer in gpu.buffers())
    # All labels remain on CPU; each original component owns its device move.
    a=cpu.compute_loss(data);b=gpu.compute_loss(data)
    assert b.device.type=='cuda' and torch.allclose(a,b.cpu(),atol=1e-6,rtol=1e-6)
    p=gpu(data['cseqs'].cuda(),data['rseqs'].cuda().long())
    mask=data['smasks'].cuda();pred=p.gather(-1,data['shft_cseqs'].cuda().unsqueeze(-1)).squeeze(-1)
    _,parts=gpu.compute_loss(data,return_components=True)
    assert torch.equal(parts['response'],F.binary_cross_entropy(pred[mask].double(),data['shft_rseqs'].cuda()[mask].double()))
