import matplotlib.pyplot as plt
import os


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def do_the_statistics(path_dir):
    dir_lst = os.listdir(path_dir)
    counter = {}
    for i in dir_lst:
        counter[i] = len(os.listdir(path_dir+i))
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.bar(counter.keys(), counter.values())
    ax.set_xlabel('Sample Labels')
    ax.set_ylabel('Quantities')
    ax.set_title(r"Samples' distributions in {}".format(path_dir))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    PATH_DIRS = [
        r"./data/alphabets/",
        r"./data/Chinese_letters/",
        r"./data/integers/"
    ]
    for path_dir in PATH_DIRS:
        do_the_statistics(path_dir)
