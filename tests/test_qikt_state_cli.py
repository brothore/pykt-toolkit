"""Ensure the real training CLI admits all preregistered state modes."""
import ast
from pathlib import Path


def test_state_modes_are_accepted_by_training_entrypoint():
    path = Path(__file__).resolve().parents[1] / 'examples/wandb_qikt_mamba_train.py'
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
               and node.args and isinstance(node.args[0], ast.Constant)
               and node.args[0].value == '--causal_adapter_mode']
    assert len(matches) == 1
    choices = ast.literal_eval(next(k.value for k in matches[0].keywords if k.arg == 'choices'))
    assert {'state_only', 'state_multiscale', 'state_interaction'} <= set(choices)
