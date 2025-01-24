

def dct_add(dct):
    dct = dct.copy()
    ex_dct = {'ex':121,'ex2':123}
    dct.update(ex_dct)
    return dct

if __name__ == '__main__':
    input_dct = {'aa':000,'ex2':131}

    store = []
    store.append(input_dct)
    new_dct = dct_add(input_dct)
    store.append(new_dct)
    a=1
