# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/1_core.ipynb (unless otherwise specified).

__all__ = ['get_test_funcs', 'print_result', 'do']

# Cell
import py
import traceback

import _pytest.config
import _pytest.tmpdir
import _pytest.main
import _pytest.runner
import _pytest.fixtures
import _pytest.python

# Cell
def get_test_funcs(**test_entries):
    cfg = _pytest.config.get_config()
    cfg.parse(args=[])

    _pytest.tmpdir.pytest_configure(config=cfg)

    ss = _pytest.main.Session.from_config(cfg)
    ss._setupstate = _pytest.runner.SetupState()
    ss._fixturemanager = _pytest.fixtures.FixtureManager(ss)

    m = _pytest.python.Module.from_parent(parent=ss, fspath=py.path.local())

    class Object(object):
        def __init__(self, **entries):
            self.__dict__.update(entries)
    m.obj = Object(**test_entries)

    test_funcs = [m]
    i = 0
    while i < len(test_funcs):
        if hasattr(test_funcs[i], 'collect'):
            test_funcs += test_funcs.pop(i).collect()
        else:
            i += 1
    return dict(enumerate(test_funcs))

# Cell
def print_result(idx, test_func, method_type):
    try:
        getattr(test_func, method_type, lambda: None)()
        print(f'=> no.{idx}  {test_func.nodeid}  {method_type}  passed\n')
    except Exception:
        print(f'=> no.{idx}  {test_func.nodeid}  {method_type}  failed\n')
        print(traceback.format_exc())

# Cell
def do(**test_entries):
    c = get_test_funcs(**test_entries)

    for i in c:
        print('')
        print_result(i, c[i], 'setup')
        print_result(i, c[i], 'runtest')