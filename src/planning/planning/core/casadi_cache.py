"""Disposable, content-addressed cache of planner functions (never solved routes)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import warnings

import casadi as ca
import numpy as np


def _json_value(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f'unsupported cache identity: {type(value).__name__}')


def function_key(inputs, *, jit=False):
    """Invalidate on numerical inputs, planner source, backend or machine ABI."""
    package = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in ('core', 'planners'):
        for path in sorted((package / directory).glob('*.py')):
            sources[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = dict(schema=1, inputs=inputs, sources=sources, casadi=ca.__version__,
                    numpy=np.__version__, platform=platform.platform(), jit=jit,
                    compiler_flags='-O2 -ffp-contract=off' if jit else None)
    raw = json.dumps(identity, sort_keys=True, allow_nan=False, default=_json_value)
    return hashlib.sha256(raw.encode()).hexdigest()


class FunctionCache:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser()

    def load(self, key):
        try:
            record = json.loads((self.directory / f'{key}.json').read_text())
            serialized = record['function']
            if record['key'] != key or record['sha256'] != hashlib.sha256(serialized.encode()).hexdigest():
                raise ValueError('function cache digest mismatch')
            return ca.Function.deserialize(serialized)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            warnings.warn(f'Ignoring unusable planner cache entry: {exc}', RuntimeWarning)
            return None

    def save(self, key, function):
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            serialized = function.serialize()
            record = dict(key=key, function=serialized,
                          sha256=hashlib.sha256(serialized.encode()).hexdigest())
            # One atomic file: concurrent seeds cannot observe a partial graph
            # or a metadata/graph pair from different writes.
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, separators=(',', ':'))
            os.replace(temporary, self.directory / f'{key}.json')
        except (OSError, RuntimeError) as exc:
            warnings.warn(f'Planner cache unavailable; using the built function: {exc}', RuntimeWarning)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def compile_function(function):
    """Optional C evaluation with the same objective and analytic gradient.

    No fast-math or native-CPU flags. Embedding permits a later process to load
    the compiled function without repeating compilation. This remains opt-in.
    """
    return function.factory(
        function.name() + '_compiled', function.name_in(), function.name_out(), {},
        dict(jit=True, compiler='shell', jit_serialize='embed',
             jit_options=dict(flags='-O2 -ffp-contract=off')),
    )
