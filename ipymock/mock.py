# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/mock.ipynb (unless otherwise specified).

__all__ = ['get_conversations', 'get_conversation', 'handle_conversation_detail', 'start_conversation',
           'generate_title', 'rename_title', 'delete_conversation', 'recover_conversation', 'clear_conversations',
           'chat_gpt_base_url', 'common', 'attrdict', 'attributize', 'delta', 'mock_create', 'mock_openai']

# Internal Cell
from queue import Queue

class Common:
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
import json, os, requests, uuid

# Cell
chat_gpt_base_url = 'http://127.0.0.1:8080'

# open the JSON file and read the access_token
with open(os.path.expanduser('~/.config/revChatGPT/config.json'), 'r') as f:
    access_token = json.load(f).get('access_token', None)

common = Common()


def get_conversations():
    response = requests.get(f'{chat_gpt_base_url}/conversations?offset=0&limit=100', headers = {'Authorization': access_token})
    return response.json()

def get_conversation(conversation_id):
    response = requests.get(f'{chat_gpt_base_url}/conversation/{conversation_id}', headers = {'Authorization': access_token})
    conversation = response.json()
    current_node = conversation['current_node']
    common.parent_message_id = current_node
    handle_conversation_detail(current_node, conversation['mapping'])
    common.exit_for_loop_channel.put(True)

def handle_conversation_detail(current_node, mapping):
    conversation_detail = mapping[current_node]
    parent_id = conversation_detail.get('parent', '')
    if parent_id != '':
        common.question_answer_map[parent_id] = conversation_detail['message']['content']['parts'][0].strip()
        handle_conversation_detail(parent_id, mapping)
    if 'message' not in conversation_detail:
        return
    message = conversation_detail['message']
    parts = message['content']['parts']
    if len(parts) > 0 and parts[0] != '':
        if message['author']['role'] == common.role_user:
            common.message_channel.put(message)

def start_conversation(content):
    if common.parent_message_id == '' or common.conversation_id == '':
        common.conversation_id = ''
        common.parent_message_id = str(uuid.uuid4())
    response = requests.post(
        f'{chat_gpt_base_url}/conversation',
        headers = {
            'Authorization': access_token,
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        data = json.dumps({
            'action': 'next',
            'messages': [{
                'id': uuid.uuid4().hex,
                'author': {
                    'role': common.role_user
                },
                'role': common.role_user,
                'content': {
                    'content_type': 'text',
                    'parts': [content]
                }
            }],
            'parent_message_id': common.parent_message_id,
            'model': common.chat_gpt_model,
            'conversation_id': common.conversation_id,
            'continue_text': ''
        }),
        stream=True
    )

    temp_conversation_id = ''
    for line in response.iter_lines():
        if not line.startswith(b'data: '):
            continue

        if line.endswith(b'[DONE]'):
            common.conversation_done_channel.put(True)
            continue

        make_conversation_response = json.loads(line.decode('utf-8')[len('data: '):])
        if make_conversation_response is None:
            continue
        parts = make_conversation_response['message']['content']['parts']
        if len(parts) > 0:
            common.response_text_channel.put(parts[0])
            yield parts[0]
        if common.conversation_id == '':
            temp_conversation_id = make_conversation_response['conversation_id']
        common.parent_message_id = make_conversation_response['message']['id']
        if make_conversation_response['message']['end_turn'] == True:
            common.conversation_done_channel.put(True)
            continue

    if common.conversation_id == '' and temp_conversation_id != '':
        common.conversation_id = temp_conversation_id
        generate_title(common.conversation_id)
    else:
        common.reload_conversations_channel.put(True)

def generate_title(conversation_id):
    requests.post(
        f'{chat_gpt_base_url}/conversation/gen_title/{conversation_id}',
        headers = {
            'Authorization': access_token,
            'Content-Type': 'application/json'
        },
        data = json.dumps({
            'message_id': common.parent_message_id,
            'model': common.chat_gpt_model
        })
    )

def rename_title(conversation_id, title):
    requests.patch(
        f'{chat_gpt_base_url}/conversation/{conversation_id}',
        headers={
            'Authorization': access_token,
            'Content-Type': 'application/json'
        },
        data = json.dumps({
            'title': title
        })
    )

def delete_conversation(conversation_id):
    requests.patch(
        f'{chat_gpt_base_url}/conversation/{conversation_id}',
        headers={
            'Authorization': access_token,
            'Content-Type': 'application/json'
        },
        data=json.dumps({
            'is_visible': False
        })
    )

def recover_conversation(conversation_id):
    requests.patch(
        f'{chat_gpt_base_url}/conversation/{conversation_id}',
        headers={
            'Authorization': access_token,
            'Content-Type': 'application/json'
        },
        data=json.dumps({
            'is_visible': True
        })
    )

def clear_conversations():
    requests.patch(f'{chat_gpt_base_url}/conversations', headers = {'Authorization': access_token}, data = {'is_visible': False})

    common.conversation_id = ''
    common.reload_conversations_channel.put(True)

# Internal Cell
# open the JSON file and read the conversation_id
with open(os.path.expanduser('~/.config/revChatGPT/config.json'), 'r') as f:
    conversation_id = json.load(f).get('conversation_id', None)

# Internal Cell
try:
    common.conversation_id = conversation_id
    get_conversation(conversation_id)
except requests.exceptions.ConnectionError as errc:
    print('Error Connecting:', errc)
except RecursionError as errr:
    print('Error Recursion:', errr)

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
        })
        res = response

def mock_create(*args, **kwargs):
    summarized_prompt = ''
    for message in kwargs['messages']:
        summarized_prompt += f"{message['role']}:\n\n{message['content']}\n\n\n"
    summarized_prompt.strip()

    if kwargs.get('stream', False):
        return delta(summarized_prompt)

    for response in start_conversation(summarized_prompt):
        pass
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
    })

# Internal Cell
import openai, pytest

# Cell
@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setattr(openai.ChatCompletion, 'create', mock_create)