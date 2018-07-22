def iob_sents(file_name):
    f = open(file_name)
    lines = f.readlines()
    f.close()
    sents = []
    sent = []
    for line in lines:
        line = line.rstrip()
        data = line.split(" ")
        if data[0] == "-DOCSTART-" or (data[0] == "" and len(sent) == 0):
            continue
        elif data[0] == "" and len(sent) > 0:
            sents.append(sent)
            sent = []
        else:
            sent.append(tuple([data[0], data[1], data[3]]))
    return sents
                                                                        
