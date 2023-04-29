# Running PyTest in Jupyter Notebooks

> iPyMock uses GPT 3.5 turbo by browser automation and <a href='https://huggingface.co/GanymedeNil/text2vec-large-chinese'>a CoSENT based model</a> to embed English and Chinese.

[![Discord Follow](https://dcbadge.vercel.app/api/server/ARTMvTQv?style=flat)](https://discord.gg/ARTMvTQv)
[![Twitter Follow](https://img.shields.io/twitter/follow/seii_saintway?style=social)](https://twitter.com/seii_saintway)

## Setup iPyMock

Get your access_token at [openai api session](https://chat.openai.com/api/auth/session) and the conversation_id in the url of chat.openai.com/c/\<conversation_id\>

```bash
mkdir -p ~/.config/ipymock

cat << EOF > ~/.config/ipymock/config.json
{
  "access_token": "<access_token>",
  "conversation_id": "<conversation_id>"
}
EOF

pip install --upgrade ipymock
```

## Using the Browser Side API in Jupyter Notebooks

```python
from ipymock.browser import start_conversation
import IPython

def ask(prompt):
    for response in start_conversation(prompt):
        IPython.display.display(IPython.core.display.Markdown(response))
        IPython.display.clear_output(wait=True)

import ipymock.browser
# if the proxy is deployed locally
ipymock.browser.common.chat_gpt_base_url = 'http://127.0.0.1:8080'
# otherwise using a third party proxy
ipymock.browser.common.chat_gpt_base_url = 'https://.../api'
# the conversation_id which is set in config.json
print(ipymock.browser.common.conversation_id)

ask('''
what is the meaning of getting patched?
''')
```

## Testing AutoGPT

This is the test function for AutoGPT.

```python
import os, sys
os.chdir(os.path.expanduser('~/Auto-GPT'))
sys.path.append(os.path.expanduser('~/Auto-GPT'))

def test_auto_gpt(
    mock_openai,
    mock_openai_embed,
    reset_embed_dimension,
):
    from autogpt.main import run_auto_gpt
    run_auto_gpt(
        continuous = True,
        continuous_limit = 10000,
        ai_settings = None,
        skip_reprompt = False,
        speak = True,
        debug = False,
        gpt3only = False,
        gpt4only = True,
        memory_type = 'local',
        browser_name = 'safari',
        allow_downloads = True,
        skip_news = True,
        workspace_directory = os.path.expanduser('~/Auto-GPT/andrew_space'),
        install_plugin_deps = True,
    )
    assert True
```

It actually run AutoGPT by pytest mock and browser automation.

```python
import ipymock
import ipymock.browser
import ipymock.llm
import pytest

ipymock.browser.common.chat_gpt_base_url = 'http://127.0.0.1:8080'
# reset conversation_id to empty to start a new chat
ipymock.browser.common.conversation_id = ''

@pytest.fixture
def reset_embed_dimension(monkeypatch):
    import autogpt.memory.local
    monkeypatch.setattr(autogpt.memory.local, 'EMBED_DIM', 1024)

ipymock.do(
    mock_openai = ipymock.browser.mock_openai,
    mock_openai_embed = ipymock.llm.mock_openai_embed,
    reset_embed_dimension = reset_embed_dimension,
    test_auto_gpt = test_auto_gpt,
)
```

This project is still under development and lack of documentation.

This is [an article](https://seii-saintway.github.io/2023/04/06/Autonomous-Agents/) I wrote that mock openai to test autonomous robots.

## The Mechanism of PyTesting the Python Testfiles

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

### PyTesting the Python Testcases within iPyNb Runtimes

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

### How to Use the Do-PyTest?

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
    


## Troubleshooting Tips of NbDev

-  Make sure you are using the latest version of nbdev with `pip install --upgrade nbdev`
-  If you are using an older version of this template, see the instructions above on how to upgrade your template. 
-  It is important for you to spell the name of your user and repo correctly in `settings.ini` or the website will not have the correct address from which to source assets like CSS for your site.  When in doubt, you can open your browser's developer console and see if there are any errors related to fetching assets for your website due to an incorrect URL generated by misspelled values from `settings.ini`.
-  If you change the name of your repo, you have to make the appropriate changes in `settings.ini`
-  After you make changes to `settings.ini`, run `nbdev_build_lib && nbdev_clean_nbs && nbdev_build_docs` to make sure all changes are propagated appropriately.
