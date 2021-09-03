def get_inst_ohlc_names(inst):
    if inst.endswith('_r'):
        return [inst[:-3] + 'o' + inst[-3 + 1:], inst[:-3] + 'h' + inst[-3 + 1:],
                inst[:-3] + 'l' + inst[-3 + 1:], inst[:-3] + 'c' + inst[-3 + 1:]]
    else:
        return [inst + 'o', inst + 'h', inst + 'l', inst + 'c']