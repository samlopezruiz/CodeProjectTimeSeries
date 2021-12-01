import matplotlib.pyplot as plt


def plot_pred(y, y_pred, steps=1):
    global fig, ax
    fig, ax = plt.subplots()
    plt.plot(y, label='observed')
    plt.plot(y_pred, label='gp')
    plt.title('pred_steps={}'.format(steps))
    ax.legend()
    plt.show()


def plot_log(df, ylabel='MDL', title=''):
    df.plot()
    plt.xlabel('GEN')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()