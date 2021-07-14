
def append_to_df(df, serie, col_name):
    df[col_name] = serie
    df[col_name].fillna(method='ffill', inplace=True)
    mask = ~df[col_name].isna()
    return df.iloc[mask.values, :].copy()

def trim_min_len(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])


def relabel(labels, map):
    if map is not None:
        for i in range(len(labels)):
            labels[i] = map[int(labels[i])]


def relabel_col(df, col, map):
    labels = df[col].to_numpy()
    relabel(labels, map=map)
    df[col] = labels
