
from utils.Expression.defs import ABOVE_BELOW_COMMANDS, ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW

# Return the valid relations for a token.
# The order is important: this is the order in which the relations are expected to occur
def get_allowed_relations(p_word_str:str)->list[str]:
    if p_word_str == '\\frac':
        allowed_relations = ['above','below']
    elif p_word_str == '\\sqrt':
        allowed_relations = ['sup','L-sup','inside']
    elif p_word_str == '\\log':
        allowed_relations = ['sub','L-sup']
    elif p_word_str == '\\stack':
        allowed_relations = ['inside']
    elif p_word_str == '\\row':
        allowed_relations = ['inside', 'below']
    elif p_word_str in ABOVE_BELOW_COMMANDS:
        allowed_relations = ['above','below','sub','sup']
    elif p_word_str in ACCENT_COMMANDS_ABOVE:
        allowed_relations = ['below','sub','sup']
    elif p_word_str in ACCENT_COMMANDS_BELOW:
        allowed_relations = ['above','sub','sup']
    else:
        allowed_relations = ['sub','sup']
    
    allowed_relations.append('right')
    return allowed_relations