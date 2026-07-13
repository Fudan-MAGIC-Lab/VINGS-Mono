import sys
from contextlib import contextmanager


_MISSING = object()


@contextmanager
def isolated_modules(stubs, reload_modules=()):
    names = set(stubs) | set(reload_modules)
    previous = {name: sys.modules.get(name, _MISSING) for name in names}
    sys.modules.update(stubs)
    for name in reload_modules:
        sys.modules.pop(name, None)

    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
