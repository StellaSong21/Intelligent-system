################################## COM ##################################
def precision(output, target):
    total = 0
    correct = 0
    i = 0
    while i < len(output):
        if output[i] == 'S':
            total += 1
            correct += 1 if target[i] == 'S' else 0
        elif output[i] == 'B':
            j = i
            i += 1
            while i < len(output) and output[i] == 'I':
                i += 1
            if i < len(output) and output[i] == 'E':
                total += 1
                correct += 1 if target[j:i + 1] == output[j:i + 1] else 0
            else:
                i -= 1
        i += 1
    return 1.0 * correct / total if total > 0 else 0.0


#########################################################################

################################## HMM ##################################
def HMM_stat(filepaths, encoding='utf8'):
    charset = dict()
    dict_start = {'B': 0, 'I': 0, 'E': 0, 'S': 0}
    dict_trans = {'B': {'I': 0, 'E': 0},
                  'I': {'I': 0, 'E': 0},
                  'E': {'B': 0, 'S': 0},
                  'S': {'S': 0, 'B': 0}}
    dict_emit = {'B': {}, 'I': {}, 'E': {}, 'S': {}}
    for path in filepaths:
        file = open(path, encoding=encoding)
        last_state = None
        for line in file:
            x = line.split()
            if (len(x)) == 0:
                continue
            # charset
            if x[0] not in charset:
                charset[x[0]] = len(charset)
            # dict_start
            dict_start[x[1]] += 1
            # dict_trans
            if last_state is not None:
                dict_trans[last_state][x[1]] += 1
            last_state = x[1]
            # dict_emit
            y = dict_emit.get(x[1])
            y[x[0]] = y.get(x[0], 0) + 1
    return charset, dict_start, dict_trans, dict_emit


#########################################################################


################################## CRF ##################################
def CRF_get_train_set(filepaths, encoding='utf8'):
    charset = set()
    observes = []
    tags = []
    for filepath in filepaths:
        observe_tmp = []
        tag_tmp = []
        file = open(filepath, encoding=encoding)
        for line in file:
            x = line.split()
            if len(x) == 0:
                continue
            charset.add(x[0])
            observe_tmp.append(x[0])
            tag_tmp.append(x[1])
            pass
        observes.append(observe_tmp)
        tags.append(tag_tmp)
    return charset, observes, tags


#########################################################################


############################# BiLSTM + CRF #############################
def BiLSTM_stat(filepaths, encoding='utf8'):
    charset = dict()
    training_data = []
    for filepath in filepaths:
        sub_sequence = []
        sub_tags = []
        file = open(filepath, encoding=encoding)
        for line in file:
            x = line.split()
            if len(x) == 0:
                if len(sub_sequence) > 0:
                    training_data.append((sub_sequence, sub_tags))
                sub_sequence = []
                sub_tags = []
                continue
            if x[0] not in charset:
                charset[x[0]] = len(charset)
            sub_sequence.append(x[0])
            sub_tags.append(x[1])
            pass
    return charset, training_data


########################################################################


################################## test ##################################
# def CRF_get_train_set(filepaths, encoding='utf8'):
#     charset = set()
#     observes = []
#     tags = []
#     for filepath in filepaths:
#         observe_tmp = []
#         tag_tmp = []
#         file = open(filepath, encoding=encoding)
#         for line in file:
#             x = line.split()
#             if len(x) == 0:
#                 continue
#             charset.add(x[0])
#             observe_tmp.append(x[0])
#             tag_tmp.append(x[1])
#             pass
#         observes.append(observe_tmp)
#         tags.append(tag_tmp)
#     return charset, observes, tags

##########################################################################
