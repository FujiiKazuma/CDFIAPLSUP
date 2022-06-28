import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
# from dataloader import load_visualize_data

import matplotlib

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 9 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 2. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"

check_num = 1
lap = 0
PU_num = 1
##

def make_paths(root_path, check_num, lap, PU_num):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/feature.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/check"))

    return ps

def visualize(check_num, lap, feature, label, savepath):
    cmap = get_cmap("tab10")
    
    X_U = feature[label != 1]
    label_U = label[label != 1]

    X_PP = feature[label == 1]
    X_UP = feature[label == 0]
    X_UN = feature[label == -1]
    X_U =  feature[label != 1]

    # 特徴量を散布図にする
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    plt.scatter(X_PP[:, 0], X_PP[:, 1], marker="*", s=70, alpha=0.6, zorder=3, label="PP", color="r")
    plt.scatter(X_UP[:, 0], X_UP[:, 1], marker="o", s=70, alpha=0.3, zorder=2, label="UP", color="violet")
    plt.scatter(X_UN[:, 0], X_UN[:, 1], marker="^", s=70, alpha=0.3, zorder=1, label="UN", color="b")
    # plt.scatter(X_U[:, 0],  X_U[:, 1],  marker="o", s=70, alpha=0.4, zorder=1, label="Unlabeled", color="mediumslateblue")

    plt.xticks(np.arange(min(feature[:, 0]), max(feature[:, 0])+0.1, 0.3))
    plt.yticks(np.arange(min(feature[:, 1]), max(feature[:, 1])+0.1, 0.3))
    
    plt.axis("off")
    # # x軸に補助目盛線を設定
    # ax.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
    #         linestyle = "--", linewidth = 1)

    # # # y軸に目盛線を設定
    # ax.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
    #         linestyle = "--", linewidth = 1)

    # plt.legend(loc='upper left')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    # plt.title(f"PCA variance = {evs}")


    save_path_vector = os.path.join(savepath, f"check_PCA_PU-ch{check_num:02}-lap{lap:02}.pdf")
    plt.savefig(save_path_vector, bbox_inches='tight', pad_inches=0.05)
    # plt.show()

    plt.close()

def main(root_path, check_num, lap, PU_num=1):
    paths = make_paths(root_path, check_num, lap, PU_num)
    ## path list ##
    # 0: feature
    # 1: label
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    matplotlib_init()

    # いろいろ読み込み
    X = np.loadtxt(paths[0])
    Xmin = X.min()
    Xmax = X.max()
    X = (X - Xmin) / (Xmax - Xmin)
    # X = X if PN = "P" else -X

    label = np.loadtxt(paths[1])[:, 3]

    # PCA
    decomp = PCA(n_components=2)
    decomp.fit(X)

    X2 = np.transpose(X)
    X_decomp = decomp.fit_transform(X)

    # 分散比の計算
    evs = explained_variance_score(X, decomp.inverse_transform(X_decomp))

    # 可視化    
    visualize(check_num, lap, X_decomp, label, paths[-1])

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")