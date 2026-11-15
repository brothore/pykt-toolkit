"""Forward controller stop requests to its active child and await safe exit."""
import signal
import subprocess


def run_cooperative(command, **kwargs):
    check = kwargs.pop('check', False)
    child = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(command, **kwargs)
        if stopping: child.send_signal(signal.SIGTERM)
        code = child.wait()
        if stopping: raise SystemExit(130)
        if check and code: raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)
