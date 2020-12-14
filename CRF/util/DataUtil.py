def get_train_set(filepaths, encoding='utf8'):
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


def get_test_set(filepath, encoding='utf8'):
    file = open(filepath, encoding=encoding)
    observes = list(file.read())
    return observes


# 返回字符集、观察集
# 字符集，set类型；观察集，list
# 考虑空格为分段，那么每一段都可以从头开始分析，因此此处在观察集和标签集中保留了' '和' '
def stat_charset1(filepaths, encoding='utf8'):
    space = []
    charset = set()
    observes = []
    tags = []
    for filepath in filepaths:
        i = -1
        space_tmp = [-1]
        observe_tmp = []
        tag_tmp = []
        file = open(filepath, encoding=encoding)
        for line in file:
            i += 1
            x = line.split()
            if len(x) == 0:
                space_tmp.append(i)
                observe_tmp.append(' ')
                tag_tmp.append(' ')
                continue
            charset.add(x[0])
            observe_tmp.append(x[0])
            tag_tmp.append(x[1])
            pass
        space_tmp.append(i + 1)
        observes.append(observe_tmp)
        tags.append(tag_tmp)
        space.append(space_tmp)
    return charset, observes, tags, space


if __name__ == '__main__':

    pass
