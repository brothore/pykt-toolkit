import importlib.util
import json
from pathlib import Path
import sys


def test_review_reentry_uses_same_identity_for_relative_and_absolute_paths(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/qikt_switch'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('discovery_reviews', scripts / 'review_discovery_rounds.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = tmp_path / 'batch'
    job = batch / 'jobs/D1'
    job.mkdir(parents=True)
    (batch / 'registration.json').write_text(json.dumps({'families': ['D1']}))
    (job / 'protocol.json').write_text(json.dumps({'family': 'D1', 'rounds': [{}]}))
    (job / 'state.json').write_text(json.dumps({'status': 'completed'}))
    result = batch / 'D1/round1'
    result.mkdir(parents=True)
    (result / 'balance.json').write_text(json.dumps({'exploratory_candidate_pass': False}))
    monkeypatch.setattr(module, 'review', lambda *args: ('review\n', 'stop'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['review', '--root', 'batch', '--output-dir', 'out', '--once'])
    module.main()
    report = tmp_path / 'out/batch/D1/round1/review.md'
    before = report.stat().st_mtime_ns
    monkeypatch.setattr(sys, 'argv', ['review', '--root', str(batch), '--output-dir', str(tmp_path / 'out'), '--once'])
    module.main()
    assert report.stat().st_mtime_ns == before
    identity = json.loads(report.with_name('source.json').read_text())
    assert all(Path(key).is_absolute() for key in identity)
