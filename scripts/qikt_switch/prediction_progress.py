"""Atomic prediction chunks and cooperative stop, independent of model code."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


class PredictionProgress:
    def __init__(self, output, identity, resume=False):
        self.root = Path(output) / 'prediction_progress'; self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.root / 'state.json'; self.stopping = False
        if self.path.exists():
            if not resume: raise ValueError('Prediction progress exists; use --resume')
            self.state = json.loads(self.path.read_text())
            if self.state['identity'] != identity: raise ValueError('Prediction inputs/code/configuration changed')
        else: self.state = {'identity': identity, 'next_window': 0, 'chunks': [], 'history': []}
        self.state['status'] = 'running'
        self.state['history'].append({'pid': os.getpid(), 'event': 'resume' if resume else 'start'})
        atomic_json(self.path, self.state)
        # Check committed chunks before advancing the sampler.
        for chunk in self.state['chunks']:
            if digest(self.root / chunk['file']) != chunk['sha256']: raise ValueError('Prediction chunk corrupted')

    def flush(self, rows, next_window):
        if next_window == self.state['next_window']: return
        name = f'chunk_{len(self.state["chunks"]):06d}.json'; path = self.root / name
        atomic_json(path, rows)
        self.state['chunks'].append({'file': name, 'sha256': digest(path), 'next_window': next_window})
        self.state['next_window'] = next_window
        atomic_json(self.path, self.state)

    def records(self):
        return [row for chunk in self.state['chunks'] for row in json.loads((self.root / chunk['file']).read_text())]

    def mark(self, status):
        self.state['status'] = status; atomic_json(self.path, self.state)

    @contextmanager
    def signals(self):
        def stop(*_): self.stopping = True
        previous = {s: signal.signal(s, stop) for s in [signal.SIGINT, signal.SIGTERM]}
        try: yield
        except SystemExit: self.mark('stopped'); raise
        except BaseException: self.mark('failed'); raise
        finally:
            for s, handler in previous.items(): signal.signal(s, handler)
            if self.state['status'] in {'stopped', 'failed'}: self.close()

    def close(self):
        self.lock.close()
