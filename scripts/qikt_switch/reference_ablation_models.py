"""Load frozen reference sources and apply explicit, isolated interventions."""
import importlib.util
import inspect
import textwrap
import types
from pathlib import Path
import torch


class ZeroResidual(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x)


def load_reference(source,config,weights=None):
    name=config['params']['model_name'];key='pykt.models._advanced_reference_'+name
    spec=importlib.util.spec_from_file_location(key,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.device=torch.device('cuda');cls=getattr(module,{'mtkt':'MTKT','lefokt_akt':'LEFOKT_AKT','ukt':'UKT'}[name])
    params=inspect.signature(cls.__init__).parameters;settings={k:v for k,v in config['model_config'].items() if k in params};cfg=config['data_config']
    settings.update(n_question=cfg['num_c'],n_pid=cfg['num_q'],emb_type=config['params']['emb_type'])
    for k in ['num_rgap','num_sgap','num_pcount']:
        if k in params:settings[k]=cfg[k]
    model=cls(**settings).cuda()
    if config.get('reference_ablation_mode','full')!='full':intervene(model,module,config['reference_ablation_mode'])
    if weights:model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    return model,module


def replace_forward(model,module,replacements):
    source=textwrap.dedent(inspect.getsource(type(model).forward))
    for old,new in replacements:
        old=old.replace('\n                    ','\n                ')
        if source.count(old)!=1:raise ValueError('Unexpected reference forward: '+old)
        source=source.replace(old,new)
    namespace={};exec(compile(source,'<registered-reference-intervention>','exec'),module.__dict__,namespace)
    model.forward=types.MethodType(namespace['forward'],model)
    return source


def intervene(model,module,mode):
    if mode=='full':return None
    if mode.endswith('_no_alibi'):
        intervene(model,module,'no_alibi')
        return intervene(model,module,mode[:-len('_no_alibi')])
    if mode in ['no_query_rasch','no_both_rasch','additive_rasch']:
        query='q_embed_data = q_embed_data + pid_embed_data * \\\n            q_embed_diff_data'
        separate='qa_embed_data = qa_embed_data + pid_embed_data * \\\n                    qa_embed_diff_data'
        combined='qa_embed_data = qa_embed_data + pid_embed_data * \\\n                    (qa_embed_diff_data+q_embed_diff_data)'
        if mode=='additive_rasch':
            # Keep the same trainable scalar PID and KC/response directions.
            # Subtract frozen initial directions so both models start identically.
            model.register_buffer('rasch_initial_q_weight',model.q_embed_diff.weight.detach().clone())
            model.register_buffer('rasch_initial_qa_weight',model.qa_embed_diff.weight.detach().clone())
            qdelta='(q_embed_diff_data - torch.nn.functional.embedding(q_data, self.rasch_initial_q_weight))'
            adelta='(qa_embed_diff_data - torch.nn.functional.embedding(target, self.rasch_initial_qa_weight))'
            replacements=[(query,'q_embed_data = q_embed_data + pid_embed_data + '+qdelta),
                (separate,'qa_embed_data = qa_embed_data + pid_embed_data + '+adelta),
                (combined,'qa_embed_data = qa_embed_data + pid_embed_data + '+adelta+' + '+qdelta)]
        else:
            replacements=[(query,'q_embed_data = q_embed_data')]
            if mode=='no_both_rasch':
                replacements.extend([(separate,'qa_embed_data = qa_embed_data'),
                    (combined,'qa_embed_data = qa_embed_data')])
        return replace_forward(model,module,replacements)
    if mode=='no_alibi':
        heads=[m for m in model.modules() if hasattr(m,'bias') and isinstance(m.bias,torch.Tensor) and m.bias.ndim==4]
        if not heads:raise ValueError('No active ALiBi bias')
        for head in heads:head.bias=torch.zeros_like(head.bias)
    elif mode=='no_cic':
        blocks=[m for m in model.modules() if hasattr(m,'cic')]
        if not blocks:raise ValueError('No CIC blocks')
        for block in blocks:block.cic=ZeroResidual()
    elif mode=='no_time':
        return replace_forward(model,module,[('temb = self.time_emb(r_gaps, s_gaps, pcounts)','temb = torch.zeros_like(q_embed_data)'),('t_out = self.model2(temb, qa_embed_data)','t_out = torch.zeros_like(q_embed_data)'),('w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_out))','w = torch.ones_like(d_output)')])
    elif mode=='equal_gate':
        return replace_forward(model,module,[('w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_out))','w = torch.full_like(d_output, 0.5)')])
    elif mode=='no_rasch':
        with torch.no_grad():model.difficult_param.weight.zero_()
        model.difficult_param.weight.requires_grad_(False)
    elif mode=='no_interaction_rasch':
        return replace_forward(model,module,[('qa_embed_data = qa_embed_data + pid_embed_data * \\\n                    qa_embed_diff_data','qa_embed_data = qa_embed_data'),('qa_embed_data = qa_embed_data + pid_embed_data * \\\n                    (qa_embed_diff_data+q_embed_diff_data)','qa_embed_data = qa_embed_data')])
    else:raise ValueError('Unknown intervention '+mode)
    return None


def predict(model,data,gaps=None):
    if model.model_name=='mtkt':return model(data,gaps)
    if model.model_name=='ukt':return model(data)
    c=torch.cat((data['cseqs'][:,:1],data['shft_cseqs']),1).long();r=torch.cat((data['rseqs'][:,:1],data['shft_rseqs']),1).long();q=torch.cat((data['qseqs'][:,:1],data['shft_qseqs']),1).long()
    return model(c,r,q)[0]
