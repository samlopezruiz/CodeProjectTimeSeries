import seaborn as sns
import matplotlib.pyplot as plt


def plot_corr_df(df):
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()