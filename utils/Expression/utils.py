
from utils.Expression.defs import ABOVE_BELOW_COMMANDS, ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW


def get_allowed_relations(p_word_str:str)->list[str]:
    if p_word_str == '\\frac':
        allowed_relations = ['above','below']
    elif p_word_str == '\\sqrt':
        allowed_relations = ['L-sup','inside','sup']
    elif p_word_str == '\\lognl':
        allowed_relations = ['L-sup','sup']
    elif p_word_str == '\\stack':
        allowed_relations = ['inside']
    elif p_word_str == '\\row':
        allowed_relations = ['inside', 'below']
    elif p_word_str in ABOVE_BELOW_COMMANDS:
        allowed_relations = ['above','below','sup','sub']
    elif p_word_str in ACCENT_COMMANDS_ABOVE:
        allowed_relations = ['below','sup','sub']
    elif p_word_str in ACCENT_COMMANDS_BELOW:
        allowed_relations = ['above','sup','sub']
    else:
        allowed_relations = ['sup','sub']
    
    return allowed_relations