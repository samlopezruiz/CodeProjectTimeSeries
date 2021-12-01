import numpy as np


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


if __name__ == '__main__':
    #%%

    X = np.full((100, 3), 2).astype(int)

    for i, j in iter_from_X_lengths(X, None):
        print(i, j)