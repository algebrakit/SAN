from typing import Optional, List
from .base import LatexItem
from .expression import Expression
from .constructs import AccentConstruct, RowConstruct, StackConstruct, Symbol, LogConstruct, FractionConstruct, SqrtConstruct, AboveBelowConstruct
from .defs import ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW, ABOVE_BELOW_COMMANDS

def parse_gtd(gtd_list) -> Optional[Expression]:
    """
    Parses a GTD (Gene Tree Description) string into its components.

    Args:
        gtd_list (list): The GTD list to parse.
        gtd_tuple:
          - [symbol, id, parent_id, parent_symbol] if parent is a symbol
          - [symbol, id, parent_id, region] if parent is a construct (e.g., '\\frac', '\\underline', etc)

    Returns:
        Expression: An Expression object representing the parsed GTD list.
    """
    expr = convert(1, gtd_list)
    return expr

 

def convert(nodeid, gtd_list) -> Optional[Expression]:
    isparent = False
    child_list = []
    items: List[LatexItem] = []
    item = None
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0],gtd_list[i][1],gtd_list[i][3]])
    if not isparent:
        items.append(Symbol(gtd_list[nodeid][0]))
    else:
        if gtd_list[nodeid][0] == '\\frac':
            above, below = None, None
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'above':
                    above = convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'below':
                    below = convert(child_list[i][1], gtd_list)
            item = FractionConstruct(construct_type='\\frac', above=above, below=below)

        elif gtd_list[nodeid][0] == '\\sqrt':
            inside, l_sup = None, None
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'inside':
                    inside = convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2].lower() in ['l-sup']:
                    l_sup = convert(child_list[i][1], gtd_list)
            if inside is None:
                inside = Expression.fromLatex(' ')
            if l_sup is None:
                item = SqrtConstruct(construct_type='\\sqrt', inside=inside)
            else:
                item = SqrtConstruct(construct_type='\\sqrt', inside=inside, l_sup=l_sup)
        elif gtd_list[nodeid][0] == '\\log':
            l_sup, sup = None, None
            for i in range(len(child_list)):
                if child_list[i][2].lower() in ['l-sup']:
                    l_sup = convert(child_list[i][1], gtd_list)
                if child_list[i][2].lower() in ['sup']:
                    sup = convert(child_list[i][1], gtd_list)
            if l_sup is None:
                item = Symbol(value='\\log', sup=sup)
            else:
                item = LogConstruct(construct_type='\\lognl', l_sup=l_sup, sup=sup)

        elif gtd_list[nodeid][0] == '\\stack':
            inside = None
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'inside':
                    inside = convert(child_list[i][1], gtd_list)
            if inside is None:
                inside = Expression.fromLatex(r'\row { }')
            item = StackConstruct(construct_type='\\stack', inside=inside)

        elif gtd_list[nodeid][0] == '\\row':
            inside, below = None, None
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'inside':
                    inside = convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'below':
                    below = convert(child_list[i][1], gtd_list)
            if inside is None:
                inside = Expression.fromLatex(' ')
            item = RowConstruct(inside=inside, below=below)

        elif gtd_list[nodeid][0] in ACCENT_COMMANDS_ABOVE or gtd_list[nodeid][0] in ACCENT_COMMANDS_BELOW:
            is_above = gtd_list[nodeid][0] in ACCENT_COMMANDS_ABOVE
            child, sub, sup = None, None, None
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'above' or child_list[i][2].lower() == 'below':
                    child = convert(child_list[i][1], gtd_list)
            if child is None:
                child = Expression.fromLatex(' ')
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'sub':
                    sub = convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2].lower() == 'sup':
                    sup = convert(child_list[i][1], gtd_list)
            item = AccentConstruct(construct_type=gtd_list[nodeid][0], is_above=is_above, child=child, sub=sub, sup=sup)

        elif gtd_list[nodeid][0] in ABOVE_BELOW_COMMANDS:
            above, below = None, None
            for i in range(len(child_list)):
                if child_list[i][2].lower() in ['above', 'sup']:
                    above = convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2].lower() in ['below', 'sub']:
                    below = convert(child_list[i][1], gtd_list)
            item = AboveBelowConstruct(construct_type=gtd_list[nodeid][0], above=above, below=below)

        else:
            symbol = Symbol(gtd_list[nodeid][0])
            for i in range(len(child_list)):
                if child_list[i][2].lower() in ['sub','below']:
                    symbol.sub = convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2].lower() in ['sup','above']:
                    symbol.sup = convert(child_list[i][1], gtd_list)
            item = symbol

        items.append(item)

        for i in range(len(child_list)):
            if child_list[i][2].lower() == 'right':
                right_expr = convert(child_list[i][1], gtd_list)
                if right_expr is not None:
                    items.extend(right_expr.get_children())

    if len(items) == 0:
        return None 
    return Expression(items)

