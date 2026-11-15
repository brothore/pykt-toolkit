import hashlib
import json
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from reuse_identical_reference_questions import main


def test_reuse_rejects_equal_values_with_different_inference_dtype(tmp_path,monkeypatch):
    source=tmp_path/'source';target=tmp_path/'target';source.mkdir();target.mkdir();raw=tmp_path/'raw.csv';raw.write_text('raw');modelsource=tmp_path/'model.py';modelsource.write_text('source')
    cfg={'params':{'emb_type':'qid'}}
    for folder in [source,target]:(folder/'config.json').write_text(json.dumps(cfg))
    torch.save({'weight':torch.ones(2,dtype=torch.float32)},source/'qid_model.ckpt');torch.save({'weight':torch.ones(2,dtype=torch.float64)},target/'qid_model.ckpt')
    srcspec={'checkpoint_dir':str(source),'output':str(source),'raw_file':str(raw),'source':str(modelsource),'batch_size':64};dstspec={**srcspec,'checkpoint_dir':str(target),'output':str(target)}
    a=tmp_path/'src.json';b=tmp_path/'dst.json';a.write_text(json.dumps(srcspec));b.write_text(json.dumps(dstspec));(source/'validation_question_results.json').write_text('{}');monkeypatch.setattr(sys,'argv',['reuse','--source-spec',str(a),'--target-spec',str(b)])
    with pytest.raises(ValueError,match='parameter tensors'):main()
    assert not (target/'validation_question_results.json').exists()
