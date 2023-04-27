# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/2_browser.ipynb (unless otherwise specified).

__all__ = ['common', 'get_conversations', 'get_conversation', 'handle_conversation_detail', 'start_conversation',
           'generate_title', 'rename_title', 'delete_conversation', 'recover_conversation', 'clear_conversations',
           'attrdict', 'attributize', 'delta', 'mock_create', 'chat_delta', 'mock_chat_create', 'mock_openai']

# Internal Cell
from queue import Queue

class Common:
    chat_gpt_base_url = 'http://127.0.0.1:8080'
    access_token = None

    chat_gpt_model = 'gpt-3.5-turbo'
    role_user = 'user'
    role_assistant = 'assistant'

    question_answer_map = {}
    message_channel = Queue()
    exit_for_loop_channel = Queue()
    response_text_channel = Queue()
    conversation_done_channel = Queue()
    parent_message_id = ''
    conversation_id = ''
    reload_conversations_channel = Queue()

# Internal Cell
import json, os, requests, sys, time, uuid

# Cell
common = Common()

# open the JSON file and read the access_token and conversation_id
with open(os.path.expanduser('~/.config/ipymock/config.json'), 'r') as f:
    config = json.load(f)
    common.access_token = config.get('access_token', None)
    common.conversation_id = config.get('conversation_id', None)

# Cell
def get_conversations():
    response = requests.get(f'{common.chat_gpt_base_url}/conversations?offset=0&limit=100', headers = {'Authorization': common.access_token})
    return response.json()

def get_conversation(conversation_id):
    response = requests.get(f'{common.chat_gpt_base_url}/conversation/{conversation_id}', headers = {'Authorization': common.access_token})
    conversation = response.json()
    current_node = conversation['current_node']
    try:
        handle_conversation_detail(current_node, conversation['mapping'])
    except RecursionError as errr:
        sys.stderr.write(f'Error Recursing: {errr}\n')
    common.exit_for_loop_channel.put(True)
    return current_node

def handle_conversation_detail(current_node, mapping):
    conversation_detail = mapping[current_node]
    parent_id = conversation_detail.get('parent', '')
    if parent_id != '':
        handle_conversation_detail(parent_id, mapping)
        common.question_answer_map[parent_id] = conversation_detail['message']['content']['parts'][0].strip()
    if 'message' not in conversation_detail:
        return
    message = conversation_detail['message']
    parts = message['content']['parts']
    if len(parts) > 0 and parts[0] != '' and message['author']['role'] == common.role_user:
        common.message_channel.put(message)

def start_conversation(content):
    if common.conversation_id != '' and common.parent_message_id == '':
        try:
            common.parent_message_id = get_conversation(common.conversation_id)
        except requests.exceptions.ConnectionError as errc:
            sys.stderr.write(f'Error Connecting: {errc}\n')

    if common.conversation_id == '' or common.parent_message_id == '':
        common.conversation_id = ''
        common.parent_message_id = str(uuid.uuid4())

    post_data = {
        'action': 'next',
        'messages': [{
            'id': str(uuid.uuid4()),
            'author': {
                'role': common.role_user,
            },
            'role': common.role_user,
            'content': {
                'content_type': 'text',
                'parts': [content],
            },
        }],
        'model': common.chat_gpt_model,
        'continue_text': '',
    }
    if common.conversation_id != '':
        post_data['conversation_id'] = common.conversation_id
    if common.parent_message_id != '':
        post_data['parent_message_id'] = common.parent_message_id

    response = requests.post(
        f'{common.chat_gpt_base_url}/conversation',
        headers = {
            'Authorization': common.access_token,
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        data = json.dumps(post_data),
        stream=True
    )

    temp_conversation_id = ''
    for line in response.iter_lines():
        if not line.startswith(b'data: '):
            continue

        if line.endswith(b'[DONE]'):
            common.conversation_done_channel.put(True)
            continue

        try:
            make_conversation_response = json.loads(line.decode('utf-8')[len('data: '):])
        except json.decoder.JSONDecodeError as err:
            sys.stderr.write(f'Error JSON Decoding: line = {line}\n')
            continue
        if make_conversation_response is None:
            continue
        try:
            parts = make_conversation_response['message']['content']['parts']
        except TypeError as err:
            sys.stderr.write(f'TypeError: {err}\nline = {line}\n')
            continue
        if len(parts) > 0:
            common.response_text_channel.put(parts[0])
            yield parts[0]
        if common.conversation_id == '':
            temp_conversation_id = make_conversation_response['conversation_id']
        common.parent_message_id = make_conversation_response['message']['id']
        if make_conversation_response['message']['end_turn'] == True:
            common.conversation_done_channel.put(True)
            continue

    if response.status_code >= 400:
        sys.stderr.write(f'Error Status Code: {response.status_code}\n')
    response.raise_for_status()

    if common.conversation_id == '' and temp_conversation_id != '':
        common.conversation_id = temp_conversation_id
        generate_title(common.conversation_id)
    else:
        common.reload_conversations_channel.put(True)

def generate_title(conversation_id):
    requests.post(
        f'{common.chat_gpt_base_url}/conversation/gen_title/{conversation_id}',
        headers = {
            'Authorization': common.access_token,
            'Content-Type': 'application/json'
        },
        data = json.dumps({
            'message_id': get_conversation(conversation_id),
            'model': common.chat_gpt_model
        })
    )

def rename_title(conversation_id, title):
    requests.patch(
        f'{common.chat_gpt_base_url}/conversation/{conversation_id}',
        headers={
            'Authorization': common.access_token,
            'Content-Type': 'application/json'
        },
        data = json.dumps({
            'title': title
        })
    )

def delete_conversation(conversation_id):
    requests.patch(
        f'{common.chat_gpt_base_url}/conversation/{conversation_id}',
        headers={
            'Authorization': common.access_token,
            'Content-Type': 'application/json'
        },
        data=json.dumps({
            'is_visible': False
        })
    )

def recover_conversation(conversation_id):
    requests.patch(
        f'{common.chat_gpt_base_url}/conversation/{conversation_id}',
        headers={
            'Authorization': common.access_token,
            'Content-Type': 'application/json'
        },
        data=json.dumps({
            'is_visible': True
        })
    )

def clear_conversations():
    requests.patch(f'{common.chat_gpt_base_url}/conversations', headers = {'Authorization': common.access_token}, data = {'is_visible': False})

    common.conversation_id = ''
    common.parent_message_id = ''
    common.reload_conversations_channel.put(True)

# Internal Cell
import random, string

# Cell
class attrdict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

def attributize(obj):
    '''Add attributes to a dictionary and its sub-dictionaries.'''
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = attributize(obj[key])
        return attrdict(obj)
    if isinstance(obj, list):
        return [attributize(item) for item in obj]
    return obj

def delta(prompt):
    id = ''.join(
        random.choices(string.ascii_letters + string.digits, k = 29)
    )
    res = ''
    for response in start_conversation(prompt):
        yield attributize({
            'choices': [
                {
                    'index': 0,
                    'logprobs': None,
                    'text': response[len(res):],
                }
            ],
            'id': f'cmpl-{id}',
        })
        res = response

def mock_create(*args, **kwargs):
    prompts = []
    if isinstance(kwargs['prompt'], str):
        prompts = [kwargs['prompt']]
    if isinstance(kwargs['prompt'], list):
        prompts = kwargs['prompt']
    prompts = [prompt.strip() for prompt in prompts]

    if kwargs.get('stream', False):
        return delta('\n'.join(prompts))

    choices = []
    for prompt in prompts:
        response = ''
        wait_second = 1
        while True:
            try:
                for response in start_conversation(prompt):
                    pass
            except requests.exceptions.HTTPError as err:
                sys.stderr.write(
                    f'{err}\n'
                    f'response = {repr(response)}\n'
                    f'Retrying...\n'
                )
                status_code = err.response.status_code
                if status_code == 413:
                    # todo: split the prompt
                    break
                # if status_code == 429:
                #     break
                if status_code >= 400 and status_code != 500:
                    time.sleep(wait_second)
                    wait_second *= 2
                    continue
                break
            if response == '':
                sys.stderr.write(
                    f'Error Responding: response = {repr(response)}\n'
                    f'Retrying...\n'
                )
                time.sleep(wait_second)
                wait_second *= 2
                continue
            break
        choices.append({
            'finish_reason': 'stop',
            'index': 0,
            'logprobs': None,
            'text': response,
        })
    id = ''.join(
        random.choices(string.ascii_letters + string.digits, k = 29)
    )
    return attributize({
        'choices': choices,
        'id': f'cmpl-{id}',
        'usage': {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        },
    })

def chat_delta(prompt):
    id = ''.join(
        random.choices(string.ascii_letters + string.digits, k = 29)
    )
    res = ''
    for response in start_conversation(prompt):
        yield attributize({
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'content': response[len(res):],
                    }
                }
            ],
            'id': f'chatcmpl-{id}',
        })
        res = response

def mock_chat_create(*args, **kwargs):
    summarized_prompt = ''
    for message in kwargs['messages']:
        summarized_prompt += f"{message['role']}:\n\n{message['content']}\n\n\n"
    summarized_prompt.strip()

    if kwargs.get('stream', False):
        return chat_delta(summarized_prompt)

    response = ''
    wait_second = 1
    while True:
        try:
            for response in start_conversation(summarized_prompt):
                pass
        except requests.exceptions.HTTPError as err:
            sys.stderr.write(
                f'{err}\n'
                f'response = {repr(response)}\n'
                f'Retrying...\n'
            )
            status_code = err.response.status_code
            if status_code == 413:
                # todo: split the prompt
                break
            # if status_code == 429:
            #     break
            if status_code >= 400 and status_code != 500:
                time.sleep(wait_second)
                wait_second *= 2
                continue
            break
        if response == '':
            sys.stderr.write(
                f'Error Responding: response = {repr(response)}\n'
                f'Retrying...\n'
            )
            time.sleep(wait_second)
            wait_second *= 2
            continue
        break
    id = ''.join(
        random.choices(string.ascii_letters + string.digits, k = 29)
    )
    return attributize({
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'message': {
                    'content': response,
                    'role': 'assistant',
                }
            }
        ],
        'id': f'chatcmpl-{id}',
        'usage': {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        },
    })

# Internal Cell
import openai, pytest

# Cell
@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setattr(openai.Completion, 'create', mock_create)
    monkeypatch.setattr(openai.ChatCompletion, 'create', mock_chat_create)