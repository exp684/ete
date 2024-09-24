import matplotlib.pyplot as plt


def plot_stats2(stats_array,stats_array2,  plot_name, x_name, y_name, folder):
    plt.plot(stats_array)
    plt.plot(stats_array2, color="red", linestyle='--')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(plot_name)
    plt.savefig(folder + "/" + plot_name + ".png")
    plt.cla()
    plt.clf()


def plot_stats(stats_array,  plot_name, x_name, y_name, folder):
    plt.plot(stats_array)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(plot_name)
    plt.savefig(folder + "/" + plot_name + ".png")
    plt.cla()
    plt.clf()