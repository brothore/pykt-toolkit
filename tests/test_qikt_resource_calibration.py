import importlib.util
from pathlib import Path


def module(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts/qikt_switch'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('calibration',scripts/'calibrate_resources.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_equal_work_and_no_duplicate_batches(monkeypatch):
    calibration=module(monkeypatch)
    for c in [1,2,3,4,6,8]:
        batches=calibration.allocation(120,c)
        assert sorted(x for worker in batches for x in worker)==list(range(120))
        assert {len(worker) for worker in batches}=={120//c}


def test_recommend_lowest_concurrency_on_throughput_plateau(monkeypatch):
    calibration=module(monkeypatch)
    rows=[{'concurrency':c,'steps_per_second':rate} for c,rates in [(1,[5,5]),(2,[9.7,9.9]),(4,[10,10]),(6,[8,8])] for rate in rates]
    choice,medians=calibration.shortlist(rows)
    assert choice==2
    assert medians[4]==10
