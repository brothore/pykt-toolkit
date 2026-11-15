import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


def test_discovery_stops_at_first_candidate_and_reuses_completed_round(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/qikt_switch'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('discovery', scripts / 'run_discovery_family.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({'orirow': [1, 1], 'qidx': [1, 2], 'late_trues': [0, 1],
                          'questions': ['1', '2'], 'concepts': ['0', '0'], 'late_mean': [.4, .6]})
    base = tmp_path / 'base.tsv'
    frame.to_csv(base, sep='\t', index=False)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'family': 'fixture', 'mode': 'fixed_mixture',
        'baseline_predictions': str(base), 'candidate_predictions': str(base),
        'rounds': [{'candidate_weight': weight} for weight in [.2, .5, .8]]}))
    calls = []
    monkeypatch.setattr(module, 'compare', lambda *args, **kwargs:
                        (calls.append(1) or {'pooled_delta': 0.}, frame))
    monkeypatch.setattr(module, 'assess_balance', lambda *args:
                        {'exploratory_candidate_pass': True})
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['discovery', '--protocol', str(protocol), '--output-dir', str(output)])
    module.main()
    assert len(calls) == 1 and not (output / 'round2').exists()
    assert json.loads((output / 'summary.json').read_text())['candidate_found']
    module.main()
    assert len(calls) == 1


def test_mixture_uses_separate_anchor_but_compares_original_baseline(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/qikt_switch'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('discovery_anchor', scripts / 'run_discovery_family.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({'orirow': [1, 1], 'qidx': [1, 2], 'late_trues': [0, 1],
                          'questions': ['1', '2'], 'concepts': ['0', '0'], 'late_mean': [.4, .6]})
    paths = {}
    for name, values in [('base', [.4, .6]), ('anchor', [.3, .7]), ('other', [.2, .8])]:
        paths[name] = tmp_path / (name + '.tsv')
        frame.assign(late_mean=values).to_csv(paths[name], sep='\t', index=False)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'family': 'fixture', 'mode': 'fixed_mixture',
        'baseline_predictions': str(paths['base']), 'anchor_predictions': str(paths['anchor']),
        'candidate_predictions': str(paths['other']), 'rounds': [{'candidate_weight': .2}]}))

    def compare(base, candidate, *args, **kwargs):
        assert base.late_mean.tolist() == [.4, .6]
        assert abs(candidate.late_mean.iloc[0] - .28) < 1e-12
        assert abs(candidate.late_mean.iloc[1] - .72) < 1e-12
        return {'pooled_delta': 0.}, frame

    monkeypatch.setattr(module, 'compare', compare)
    monkeypatch.setattr(module, 'assess_balance', lambda *args: {'exploratory_candidate_pass': False})
    monkeypatch.setattr(sys, 'argv', ['discovery', '--protocol', str(protocol),
                                     '--output-dir', str(tmp_path / 'output')])
    module.main()
