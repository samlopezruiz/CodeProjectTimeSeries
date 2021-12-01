
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import networkx as nx # Used only for PROMTEHEE I visualisation (could be omitted if unnecessary)

# Print nice-looking tables
from tabulate import tabulate

from pymcdm import methods as mcdm_methods
from pymcdm import weights as mcdm_weights
from pymcdm import normalizations as norm
from pymcdm import correlations as corr
from pymcdm.helpers import rankdata, rrankdata

if __name__ == '__main__':
    data = pd.read_csv('vans.csv')

    #%%
    matrix = data[data.columns[3:]].to_numpy()
    print(matrix)

    #%%
    weights = mcdm_weights.equal_weights(matrix)
    print(weights)
    print(mcdm_weights.entropy_weights(matrix))
    print(mcdm_weights.standard_deviation_weights(matrix))

    #%%
    '''
    Functions in our library use types argument to determine which
    criteria are profit and should be maximized and which criteria are cost
    and should be minimized. 1 means profit criteria and −1 means cost criteria.
    For this particular decision problem types vector would be defined as follows:
    '''
    types = np.array([1, 1, 1, 1, 1, -1, -1, 1, -1])

    #%%
    topsis = mcdm_methods.TOPSIS()
    print(topsis(matrix, weights, types))
    print(rankdata(topsis(matrix, weights, types), reverse=True))

    #%%
    topsis_methods = {
        'minmax': mcdm_methods.TOPSIS(norm.minmax_normalization),
        'max': mcdm_methods.TOPSIS(norm.max_normalization),
        'sum': mcdm_methods.TOPSIS(norm.sum_normalization),
        'vector': mcdm_methods.TOPSIS(norm.vector_normalization),
        'log': mcdm_methods.TOPSIS(norm.logaritmic_normalization),
    }

    results = {}
    for name, function in topsis_methods.items():
        results[name] = function(matrix, weights, types)

    print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
                   headers=['Method'] + [f'A{i + 1}' for i in range(10)]))

    print(tabulate([[name, *rankdata(pref, reverse=True)] for name, pref in results.items()],
                   headers=['Method'] + [f'A{i + 1}' for i in range(10)]))


    #%%
    vikor = mcdm_methods.VIKOR()
    print(vikor(matrix, weights, types))

    #%%
    vikor_methods = {
        'VIKOR': mcdm_methods.VIKOR(),
        'minmax': mcdm_methods.VIKOR(norm.minmax_normalization),
        'max': mcdm_methods.VIKOR(norm.max_normalization),
        'sum': mcdm_methods.VIKOR(norm.sum_normalization),
        'vector': mcdm_methods.VIKOR(norm.vector_normalization),
        'log': mcdm_methods.VIKOR(norm.logaritmic_normalization),
    }

    results = {}
    for name, function in vikor_methods.items():
        results[name] = function(matrix, weights, types)

    print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
                   headers=['Method'] + [f'A{i + 1}' for i in range(10)]))

    print(tabulate([[name, *rankdata(pref)] for name, pref in results.items()],
                   headers=['Method'] + [f'A{i + 1}' for i in range(10)]))

    #%%
    preference_functions = ['usual', 'vshape', 'ushape', 'level', 'vshape_2']
    promethee_methods = {
        f'{pref}': mcdm_methods.PROMETHEE_II(preference_function=pref)
        for pref in preference_functions
    }

    p = np.random.rand(matrix.shape[1]) / 2
    q = np.random.rand(matrix.shape[1]) / 2 + 0.5
    results = {}
    for name, function in promethee_methods.items():
        results[name] = function(matrix, weights, types, p=p, q=q)

    print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
                   headers=[''] + [f'A{i + 1}' for i in range(10)]))

    print(tabulate([[name, *rrankdata(pref)] for name, pref in results.items()],
                   headers=[''] + [f'A{i + 1}' for i in range(10)]))

    #%%
    def promethee_I_visualization(Fp, Fm):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.51, 0.51)
        ax.set_ylim(-0.51, 0.51)
        ax.axis('off')

        ax.plot([0.5, 0.5], [-0.5, 0.5], 'k', linewidth=3)
        ax.plot([-0.5, -0.5], [-0.5, 0.5], 'k', linewidth=3)
        ax.plot([0, 0], [-0.5, 0.5], 'k',
                alpha=0.5, linewidth=1, linestyle='--')

        ax.text(-0.61, 0, 'Phi+', fontsize='large')
        ax.text(-0.56, 0.5, '1.0')
        ax.text(-0.56, -0.5, '0.0')

        ax.text(0.52, 0, 'Phi-', fontsize='large')
        ax.text(0.51, -0.5, '1.0')
        ax.text(0.51, 0.5, '0.0')

        for i, (fp, fm) in enumerate(zip(Fp, Fm)):
            ax.plot([-0.5, 0.5], [-0.5 + fp, 0.5 - fm], label=f'A{i + 1}')
            ax.text(-0.55, -0.5 + fp, f'A{i + 1}')
            ax.text(0.51, 0.5 - fm, f'A{i + 1}')
        plt.legend()
        plt.show()


    def promethee_II_vizualization(Fi):
        fig, ax = plt.subplots()
        ax.set_xlim(-0.51, 0.51)
        ax.set_ylim(-0.51, 0.51)
        ax.axis('off')

        ax.plot([0, 0], [-0.5, 0.5], 'k', linewidth=5)
        ax.text(0.02, 0.5, '1.0')
        ax.text(0.02, 0, '0.0')
        ax.text(0.02, -0.5, '-1.0')

        for i in np.arange(-0.5, 0.51, 0.25):
            ax.plot([-0.02, 0.02], [i, i], 'k')

        for i, fi in enumerate(Fi):
            ax.plot([-0.1, 0.1], [fi / 2, fi / 2], label=f'A{i + 1}')
            ax.text(-0.15, fi / 2, f'{fi:0.2f}')
            ax.text(0.1, fi / 2, f'A{i + 1}')
        plt.legend()
        plt.show()

    #%%
    promethee = mcdm_methods.PROMETHEE_II('usual')
    Fp, Fm = promethee(matrix, weights, types, promethee_I=True)
    promethee_I_visualization(Fp, Fm)

    #%%
    Fi = promethee(matrix, weights, types)
    ranks = rrankdata(Fi)
    print(ranks)
    promethee_II_vizualization(Fi)