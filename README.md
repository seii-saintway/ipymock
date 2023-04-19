# A Jupyter Notebook Runtime Pytest Plugin
> This pytest plugin project is created by the <a href='https://youtu.be/ZJTop5uqC2U'>NbDev Template</a>.


## Troubleshooting Tips

-  Make sure you are using the latest version of nbdev with `pip install -U nbdev`
-  If you are using an older version of this template, see the instructions above on how to upgrade your template. 
-  It is important for you to spell the name of your user and repo correctly in `settings.ini` or the website will not have the correct address from which to source assets like CSS for your site.  When in doubt, you can open your browser's developer console and see if there are any errors related to fetching assets for your website due to an incorrect URL generated by misspelled values from `settings.ini`.
-  If you change the name of your repo, you have to make the appropriate changes in `settings.ini`
-  After you make changes to `settings.ini`, run `nbdev_build_lib && nbdev_clean_nbs && nbdev_build_docs` to make sure all changes are propagated appropriately.

## Pytest the Python Testfiles

```python
import pluggy
pm = pluggy.PluginManager('pytest')

import _pytest.hookspec
pm.add_hookspecs(_pytest.hookspec)

import _pytest.main
pm.register(_pytest.main, 'main')

import _pytest.config
cfg = _pytest.config.get_config()
cfg.parse(args=[])
pm.hook.pytest_cmdline_main(config=cfg)
```

```python
import _pytest.config
cfg = _pytest.config.get_config()
cfg.parse(args=[])
cfg.pluginmanager.hook.pytest_cmdline_main(config=cfg)
```

## Pytest the Python Testcases within IPyNb Runtimes

```python
# content of test_time.py

import pytest

from datetime import datetime, timedelta


testdata = [
    (datetime(2001, 12, 12), datetime(2001, 12, 11), timedelta(1)),
    (datetime(2001, 12, 11), datetime(2001, 12, 12), timedelta(-1)),
]


def idfn(val):
    if isinstance(val, (datetime,)):
        # note this wouldn't show any hours/minutes/seconds
        return val.strftime('%Y%m%d')


@pytest.mark.parametrize('a,b,expected', testdata)
def test_timedistance_v0(a, b, expected):
    diff = a - b
    assert diff == expected


@pytest.mark.parametrize('a,b,expected', testdata, ids=['forward', 'backward'])
def test_timedistance_v1(a, b, expected):
    diff = a - b
    assert diff == expected


@pytest.mark.parametrize('a,b,expected', testdata, ids=idfn)
def test_timedistance_v2(a, b, expected):
    diff = a - b
    assert diff == expected


@pytest.mark.parametrize(
    'a,b,expected',
    [
        pytest.param(
            datetime(2001, 12, 12), datetime(2001, 12, 11), timedelta(1), id='forward'
        ),
        pytest.param(
            datetime(2001, 12, 11), datetime(2001, 12, 12), timedelta(-1), id='backward'
        ),
    ],
)
def test_timedistance_v3(a, b, expected):
    diff = a - b
    assert diff != expected
```

```python
import _pytest.config
cfg = _pytest.config.get_config()
cfg.parse(args=[])

import _pytest.main
ss = _pytest.main.Session.from_config(cfg)
import _pytest.runner
ss._setupstate = _pytest.runner.SetupState()
import _pytest.fixtures
ss._fixturemanager = _pytest.fixtures.FixtureManager(ss)

import _pytest.python
import py
m = _pytest.python.Module.from_parent(parent=ss, fspath=py.path.local())
```

```python
class Object(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
m.obj = Object(**globals())
```

```python
import _pytest.runner
c = dict(enumerate(m.collect()))
for i in c:
    print(f'idx = {i}')
    print(_pytest.runner.call_and_report(c[i], 'setup'))
    print(_pytest.runner.call_and_report(c[i], 'call'))
    print(_pytest.runner.call_and_report(c[i], 'teardown', nextitem=c.get(i+1)))
```

    idx = 0
    <TestReport '::unittests::test_timedistance_v0[a0-b0-expected0]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v0[a0-b0-expected0]' when='call' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v0[a0-b0-expected0]' when='teardown' outcome='passed'>
    idx = 1
    <TestReport '::unittests::test_timedistance_v0[a1-b1-expected1]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v0[a1-b1-expected1]' when='call' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v0[a1-b1-expected1]' when='teardown' outcome='passed'>
    idx = 2
    <TestReport '::unittests::test_timedistance_v1[forward]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v1[forward]' when='call' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v1[forward]' when='teardown' outcome='passed'>
    idx = 3
    <TestReport '::unittests::test_timedistance_v1[backward]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v1[backward]' when='call' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v1[backward]' when='teardown' outcome='passed'>
    idx = 4
    <TestReport '::unittests::test_timedistance_v2[20011212-20011211-expected0]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v2[20011212-20011211-expected0]' when='call' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v2[20011212-20011211-expected0]' when='teardown' outcome='passed'>
    idx = 5
    <TestReport '::unittests::test_timedistance_v2[20011211-20011212-expected1]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v2[20011211-20011212-expected1]' when='call' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v2[20011211-20011212-expected1]' when='teardown' outcome='passed'>
    idx = 6
    <TestReport '::unittests::test_timedistance_v3[forward]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v3[forward]' when='call' outcome='failed'>
    <TestReport '::unittests::test_timedistance_v3[forward]' when='teardown' outcome='passed'>
    idx = 7
    <TestReport '::unittests::test_timedistance_v3[backward]' when='setup' outcome='passed'>
    <TestReport '::unittests::test_timedistance_v3[backward]' when='call' outcome='failed'>
    <TestReport '::unittests::test_timedistance_v3[backward]' when='teardown' outcome='passed'>


```python
for i, f in enumerate(m.collect()):
    print(f'idx = {i}')
    f.setup()
    f.runtest()
```

### How to use the do-pytest?

```python
import pytest

@pytest.fixture
def my_fixture_1(tmpdir_factory):
    return tmpdir_factory

@pytest.fixture
def my_fixture_2(tmpdir_factory):
    return tmpdir_factory

def test_fixture(my_fixture_1, my_fixture_2):
    assert my_fixture_1 == my_fixture_2
```

```python
from ipymock import do
```

```python
do(
    my_fixture_1=my_fixture_1,
    my_fixture_2=my_fixture_2,
    test_fixture=test_fixture
)
```

    
    => no.0  nbs::test_fixture  setup  passed
    
    => no.0  nbs::test_fixture  runtest  passed
    

