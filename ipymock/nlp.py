# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/4_tools_pyknp.ipynb (unless otherwise specified).

__all__ = ['is_halfwidth', 'halfwidth_to_fullwidth', 'annotate']

# Internal Cell
from pyknp import Juman

# Cell
def is_halfwidth(text):
    """
    Determine whether the text consists entirely of halfwidth characters.
    :param text: Input text string
    :return: True if all characters are halfwidth, otherwise False
    """
    return all('\u0020' <= char <= '\u007E' or '\uFF61' <= char <= '\uFF9F' for char in text)

def halfwidth_to_fullwidth(text):
    result = ''
    for char in text:
        code = ord(char)
        if code == ord(' '):
            result += '　'
        elif ord('!') <= code <= ord('~'):
            # Convert ASCII characters in the range 0x0021 to 0x007E
            result += chr(code + 0xFEE0)
        else:
            result += char
    return result

def annotate(text):
    juman = Juman()
    for line in text.split('\n'):
        if line == '':
            yield '\n'
            continue
        mrphs = juman.analysis(line).mrph_list()
        for mrph in mrphs:
            if mrph.midasi == '\\␣':
                yield ' '
                continue
            if is_halfwidth(mrph.midasi):
                yield mrph.midasi
                continue
            if mrph.midasi == mrph.yomi:
                yield mrph.midasi
                continue
            yield f'<ruby>{mrph.midasi}<rt>{mrph.yomi}</rt></ruby>'
        yield '\n'