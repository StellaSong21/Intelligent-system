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


if __name__ == '__main__':
    pass
