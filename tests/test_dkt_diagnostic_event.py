import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
import copy,pytest,torch,numpy as np,pandas as pd
from diagnose_dkt_event import DiagnosticJoint,event_probabilities,event_fields
from diagnose_dkt_window import make_windows
from test_dkt_joint import batch,configs


def data():
    d=batch();d.update(safe_anchors=torch.tensor([[0,0],[1,1]]),safe_concepts=torch.tensor([[[0,1],[1,-1]],[[1,-1],[0,1]]]),safe_labels=torch.tensor([[.5,1.],[0.,.5]]),safe_mask=torch.ones(2,2,dtype=torch.bool));return d


def test_event_mean_queries_pre_question_state_not_teacher_forced_copy():
    d=data();m=DiagnosticJoint(2,8,0,dataset_name='assist2009',task_configs=configs('assist2009'),loss_weights={},primary_mode='event')
    y=m(d['cseqs'],d['rseqs'].long());p,mask=event_probabilities(y,d)
    assert torch.equal(p[0,0],y[0,0].double().mean())
    loss=m.compute_loss(d);expected=torch.nn.functional.binary_cross_entropy(p,d['safe_labels'].double());assert torch.equal(loss,expected)
    changed=copy.deepcopy(d);changed['rseqs'][0,1:]=1-changed['rseqs'][0,1:]
    p2,_=event_probabilities(m(changed['cseqs'],changed['rseqs'].long()),changed);assert torch.equal(p[0],p2[0])
    loss.backward();assert m.lstm_layer.weight_ih_l0.grad.abs().sum()>0


def test_original_mode_reuses_original_joint_exactly():
    from pykt.models.dkt_joint import DKTJoint
    d=data();w={'option':.1,'confidence':.0455,'profile':0}
    torch.manual_seed(42);a=DKTJoint(2,8,.2,dataset_name='nips_task34',task_configs=configs('nips_task34'),loss_weights=w)
    torch.manual_seed(42);b=DiagnosticJoint(2,8,.2,dataset_name='nips_task34',task_configs=configs('nips_task34'),loss_weights=w,primary_mode='original');rng=torch.random.get_rng_state()
    x=a.compute_loss(d);torch.random.set_rng_state(rng);assert torch.equal(x,b.compute_loss(d))


def test_event_group_cross_chunk_and_no_prefix_exclusion(tmp_path):
    from test_dkt_profile import fixture
    fixture(tmp_path,split=True);records,repeat,stats=event_fields(tmp_path/'train_valid_sequences.csv',{1},2)
    # The two-concept middle event straddles chunks and must never become
    # a partial whole-question target in either chunk.
    assert stats['excluded_partial_target_tails']==1
    assert sum(int(r['safe_mask'].sum()) for r in records)==3


def test_fast_windows_no_current_response_and_199_history_cap():
    cs=list(range(2))*105;rs=[i%2 for i in range(210)];rep=[0]*210;rep[2]=1
    frame=pd.DataFrame([{'uid':1,'concepts':','.join(map(str,cs)),'responses':','.join(map(str,rs)),'is_repeat':','.join(map(str,rep))}])
    # Ensure the repeated concept copy has the same raw event label.
    frame.loc[0,'responses']=','.join(map(str,[*rs[:2],rs[1],*rs[3:]]))
    windows=make_windows(frame)
    assert windows['late_c'].shape[1]==199 and windows['late_c'].shape[0]==10
    assert windows['times'][0]==0
