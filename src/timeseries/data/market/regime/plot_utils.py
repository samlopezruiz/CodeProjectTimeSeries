from matplotlib import pyplot as plt
import seaborn as sns


def hist_vars_state(df_plot, features, hue_var='state'):
    fig, axes = plt.subplots(len(features), 1, figsize=(15, 12))
    for j, feature in enumerate(features):
        sns.histplot(df_plot, x=feature, hue=hue_var, ax=axes[j])
    plt.tight_layout()
    plt.show()
