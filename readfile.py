def loadfile(filename):
    # File in Linux or Windows is different
    f = open(filename, 'r').read()
    raw_data = f.split('\n')
    data = []

    if raw_data[-1] == '':
        raw_data.pop()

    for i in range(len(raw_data)):
        if raw_data[i][-2:] == ' \r':
            raw_data[i] = raw_data[i][:-2]
        elif raw_data[i][-1:] == '\r':
            raw_data[i] = raw_data[i][:-1]
        elif raw_data[i][-1] == ' ':
            raw_data[i] = raw_data[i][:-1]
        else:
            pass

        temp = map(float, raw_data[i].split(' '))
        if len(temp) == 8:
            temp.append(0.0)

        data.append(temp)

    return data
