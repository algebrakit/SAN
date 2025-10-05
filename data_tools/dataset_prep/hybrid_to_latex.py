
from utils.Expression.defs import ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW

def _convert_to_gtd(lines):
    gtd_list = [ ['<sos>', 0, -1, '<sos>'] ]
    for i in range(len(lines)):
        if lines[i][1] in ['struct', '<eos>']:
            continue
        
        parent_id = lines[i][2]
        parent_type = None
        if parent_id == 0:
            parent_type = 'start'
        elif parent_id>0 and lines[parent_id-1][1] == 'struct':
            parent_id -= 1
            parent_type = lines[i][3]
        else:
            parent_type = 'right'

        gtd_list.append([lines[i][1], lines[i][0], parent_id, parent_type])

    for i in range(len(gtd_list)):
        current_id = gtd_list[i][1]
        for j in range(i+1, len(gtd_list)):
            if gtd_list[j][2] == current_id:
                gtd_list[j][2] = i

    for i in range(len(gtd_list)):
        gtd_list[i][1] = i

    return gtd_list

def _iter(nodeid, gtd_list, swapSubSup:bool):       
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0],gtd_list[i][1],gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'above':
                    return_string += ['{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'below':
                    return_string += ['{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'right':
                    return_string += _iter(child_list[i][1], gtd_list,swapSubSup)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['right','above','below']:
                    return_string += ['illegal']
        elif gtd_list[nodeid][0] == '\\stack':
            return_string = [gtd_list[nodeid][0],'{']
            rows = []
            for i in range(len(child_list)):
                if child_list[i][2] == 'below':
                    rows.append(' '.join(_iter(child_list[i][1], gtd_list,swapSubSup)))
            args = r' \\ '.join(rows)
            return_string.append(args)
            return_string.append('}')     

            for i in range(len(child_list)):
                if child_list[i][2] == 'right':
                    return_string += _iter(child_list[i][1], gtd_list,swapSubSup)

        elif gtd_list[nodeid][0] in ACCENT_COMMANDS_ABOVE or gtd_list[nodeid][0] in ACCENT_COMMANDS_BELOW:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['above', 'below']:
                    return_string += ['{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            if swapSubSup:
                for i in range(len(child_list)):
                    if child_list[i][2] == 'sub':
                        return_string += ['_','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] == 'sup':
                        return_string += ['^','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            else:
                for i in range(len(child_list)):
                    if child_list[i][2] == 'sup':
                        return_string += ['^','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] == 'sub':
                        return_string += ['_','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['right']:
                    return_string += _iter(child_list[i][1], gtd_list,swapSubSup)
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['L-sup']:
                    return_string += ['['] + _iter(child_list[i][1], gtd_list,swapSubSup) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'inside':
                    return_string += ['{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            if swapSubSup:
                for i in range(len(child_list)):
                    if child_list[i][2] in ['sub','below']:
                        return_string += ['_','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] in ['sup','above']:
                        return_string += ['^','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
            else:
                for i in range(len(child_list)):
                    if child_list[i][2] in ['sup','above']:
                        return_string += ['^','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']
                for i in range(len(child_list)):
                    if child_list[i][2] in ['sub','below']:
                        return_string += ['_','{'] + _iter(child_list[i][1], gtd_list,swapSubSup) + ['}']

            for i in range(len(child_list)):
                if child_list[i][2] in ['right']:
                    return_string += _iter(child_list[i][1], gtd_list,swapSubSup)
        return return_string

def hybrid_to_latex(nodeid, lines, swapSubSup=False): 
    gtd_list = _convert_to_gtd(lines)
    return ' '.join(_iter(nodeid, gtd_list, swapSubSup))

