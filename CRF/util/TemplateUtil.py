import re


def get_templates(filepaths, encoding='utf8'):
    templates = [[], []]
    for filepath in filepaths:
        file = open(filepath, encoding=encoding)
        for line in file:
            index, value = str2template(line)
            if index == -1:
                continue
            if value not in templates[index]:
                templates[index].append(value)
            pass
    return templates


# 因为模板名称没有意义，所以并不返回
# 处理模板，返回模板和对应位置列表
# 如果不是以U或者B开头，直接返回
# 因为都处于第一列，所以只取相对行数
def str2template(string):
    if len(string) == 0:
        return -1, None
    if string[0] not in ['B', 'U']:
        return -1, None
    string = string.strip()
    digits = re.findall(r'-?\d+', string)

    value = []
    for i in range(1, len(digits), 2):
        # value.append(list(map(int, digits[i:i + 1])))
        value.append(int(digits[i]))
    return (0 if string[0] == 'U' else 1), value
    # template = dict()
    # name = string[:3]
    # template[name] = value
    # return templates


if __name__ == '__main__':
    templates = get_templates(['../../DATASET/dataset1/template.utf8', '../../DATASET/dataset2/template.utf8'])
    print(templates)
