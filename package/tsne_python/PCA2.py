import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
# from dataloader import load_visualize_data

import matplotlib
# del matplotlib.font_manager.weight_dict['roman']

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 0.6 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 1.5 # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"

check_num = 1
lap = 0
PU_num = 1
PC_num = 1
push_direction = "N"

sr = 0.05
##


def make_paths(root_path, check_num, lap, push_direction, PU_num, PC_num):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/feature.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/P/label.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/predict.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/gradient.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/check"))

    return ps

def visualize(check_num, lap, feature, pred, label, Plabel, grad, evs, push_direction, savepath):
    if push_direction == "P":
        top_neg = feature[label != 1][np.argmax(pred[label != 1])]
    elif push_direction == "N":
        top_neg = feature[Plabel == 1][np.argmax(pred[Plabel == 1])]
    strsr = str(sr * 100)

    grad = grad[1] / grad[0]
    
    cmap = get_cmap("tab10")
    X_U = feature[label != 1]
    pred_U = pred[label != 1]
    label_U = label[label != 1]

    X_PP = feature[label == 1]
    X_UP = feature[(label == 0) & (Plabel != 1)]
    X_UN = feature[(label == -1) & (Plabel != 1)]

    # Uのうち、Pに変更されたもの(上位20%)
    X_UPtoP = feature[(label == 0) & (Plabel == 1)]
    X_UNtoP = feature[(label == -1) & (Plabel == 1)]
    # Uのうち、疑似ラベルとして選択されるもの(上位 5% or 2.5%)
    sn = int(sr * np.sum(label != 1))

    lab_pre_X_U = np.concatenate([label_U[:, np.newaxis], pred_U[:, np.newaxis], X_U], 1)
    high_index = np.argpartition(pred_U, -sn)[-sn:]
    lab_pre_X_U_high = lab_pre_X_U[high_index]
    X_UNtoPN  = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] == -1][:, -2:]
    X_UPtoPN  = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] ==  0][:, -2:]

    # 特徴量を散布図にする
    fig = plt.figure(figsize=(15, 10))
    plt.scatter(X_PP[:, 0], X_PP[:, 1], marker="*", zorder=3, alpha=0.6, label="PP", color="red")
    plt.scatter(X_UP[:, 0], X_UP[:, 1], marker="o", zorder=1, alpha=0.6, label="UP", color="purple")
    plt.scatter(X_UN[:, 0], X_UN[:, 1], marker="^", zorder=1, alpha=0.6, label="UN", color="blue")

    plt.scatter(X_UPtoP[:, 0],  X_UPtoP[:, 1],  marker="o", zorder=2, alpha=0.6, label="UP of top 20%", color="coral")
    plt.scatter(X_UNtoP[:, 0],  X_UNtoP[:, 1],  marker="^", zorder=2, alpha=0.6, label="UN of top 20%", color="teal")
    plt.scatter(X_UPtoPN[:, 0], X_UPtoPN[:, 1], marker="o", zorder=2, alpha=0.6, label=f"UP of top {strsr}%", color="olive")
    plt.scatter(X_UNtoPN[:, 0], X_UNtoPN[:, 1], marker="^", zorder=2, alpha=0.6, label=f"UN of top {strsr}%", color="orange")

    # 重みをtop_negを通る軸にする
    left, right = plt.xlim()
    up, low = plt.ylim()
    y1 = lambda x: grad * x + top_neg[1] - grad * top_neg[0]
    y2 = lambda x: -1/grad * x + top_neg[1] - -1/grad * top_neg[0]
    # plt.plot(feature[:, 0], y1(feature[:, 0]), color="black", alpha=0.5)
    # plt.plot(feature[:, 0], y2(feature[:, 0]), color="orange", alpha=0.5)
    plt.plot(feature[:, 0], np.full(feature.shape[0], top_neg[1]), color="black", alpha=0.5)
    plt.plot(np.full(feature.shape[0], top_neg[0]), feature[:, 1], color="orange", alpha=0.5)
    plt.xlim(left, right)
    plt.ylim(up, low)

    plt.axis("off")
    # plt.legend(loc='upper left')
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
    # plt.title(f"PCA variance = {evs}")

    save_path_vector = os.path.join(savepath, f"check_PCA_PC_{push_direction}-ch{check_num:02}-lap{lap:02}.pdf")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)
    # plt.show()

    plt.close()

def visualize2(check_num, lap, feature, pred, label, Plabel, grad, evs, push_direction, savepath):
    if push_direction == "P":
        top_neg = feature[label != 1][np.argmax(pred[label != 1])]
    elif push_direction == "N":
        top_neg = feature[Plabel == 1][np.argmax(pred[Plabel == 1])]
    strsr = str(sr * 100)
    
    X_U = feature[label != 1]
    pred_U = pred[label != 1]
    label_U = label[label != 1]

    X_PP = feature[label == 1]
    X_UP = feature[(label == 0) & (Plabel != 1)]
    X_UN = feature[(label == -1) & (Plabel != 1)]

    # Uのうち、Pに変更されたもの(上位20%)
    X_UPtoP = feature[(label == 0) & (Plabel == 1)]
    X_UNtoP = feature[(label == -1) & (Plabel == 1)]
    border20 = feature[(label != 1) & (Plabel == 1)][:, 0].min()
    # Uのうち、疑似ラベルとして選択されるもの(上位 5% or 2.5%)
    sn = int(sr * np.sum(label != 1))

    lab_pre_X_U = np.concatenate([label_U[:, np.newaxis], pred_U[:, np.newaxis], X_U], 1)
    high_index = np.argpartition(pred_U, -sn)[-sn:]
    lab_pre_X_U_high = lab_pre_X_U[high_index]
    X_UNtoPN  = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] == -1][:, -2:]
    X_UPtoPN  = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] ==  0][:, -2:]
    border5 = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] != 1][:, -2].min()

    # 特徴量を散布図にする
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    # 5%, 20%の領域を塗りつぶし
    featxmax, featymin, featymax = feature[:, 0].max()+0.05, feature[:, 1].min()-0.05, feature[:, 1].max()+0.05
    fillx = [border20,border5, border5, border20]
    filly = [featymin, featymin, featymax, featymax]
    if push_direction == "P":
        plt.fill(fillx, filly, color="orange", alpha=0.2)
    fillcolor = "red" if push_direction == "P" else "blue"
    fillx = [border5, featxmax, featxmax, border5]
    plt.fill(fillx, filly, color=fillcolor, alpha=0.2)

    # プロット
    plt.scatter(X_PP[:, 0], X_PP[:, 1], marker="*", zorder=3, alpha=0.3, label="PP", color="red")
    plt.scatter(X_UP[:, 0], X_UP[:, 1], marker="o", zorder=1, alpha=0.6, label="UP", color="purple")
    plt.scatter(X_UN[:, 0], X_UN[:, 1], marker="^", zorder=1, alpha=0.4, label="UN", color="blue")

    plt.scatter(X_UPtoP[:, 0],  X_UPtoP[:, 1],  marker="o", zorder=2, alpha=0.6, color="coral")
    plt.scatter(X_UNtoP[:, 0],  X_UNtoP[:, 1],  marker="^", zorder=2, alpha=0.6, color="teal")
    plt.scatter(X_UPtoPN[:, 0], X_UPtoPN[:, 1], marker="o", zorder=2, alpha=0.6, color="olive")
    plt.scatter(X_UNtoPN[:, 0], X_UNtoPN[:, 1], marker="^", zorder=2, alpha=0.6, color="orange")

    # 5%, 20%の境界を通る軸
    # left, right = plt.xlim()
    # up, low = plt.ylim()
    # if push_direction == "P":
    #     plt.plot(np.full(feature.shape[0], border20), feature[:, 1], color="orange", alpha=0.5)
    # plt.plot(np.full(feature.shape[0], border5),  feature[:, 1], color=fillcolor, alpha=0.5)
    # plt.xlim(left, right)
    # plt.ylim(up, low)

    # plt.axis("off")
    dir = "Top" if push_direction == "P" else "Bottom"
    ax.set_xlabel(f"Ranking by P-classification({dir}-push)")
    ax.set_ylabel("First principal component", size=20)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.5)

    save_path_vector = os.path.join(savepath, f"check_PCA_PC_{push_direction}-ch{check_num:02}-lap{lap:02}.pdf")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0.1)
    # plt.show()

    plt.close()

def visualize3(check_num, lap, feature, pred, label, Plabel, grad, evs, push_direction, savepath):
    if push_direction == "P":
        top_neg = feature[label != 1][np.argmax(pred[label != 1])]
    elif push_direction == "N":
        top_neg = feature[Plabel == 1][np.argmax(pred[Plabel == 1])]
    strsr = str(sr * 100)
    
    X_U = feature[label != 1]
    pred_U = pred[label != 1]
    label_U = label[label != 1]

    X_PP = feature[label == 1]
    X_UP = feature[(label == 0) & (Plabel != 1)]
    X_UN = feature[(label == -1) & (Plabel != 1)]

    # Uのうち、Pに変更されたもの(上位20%)
    X_UPtoP = feature[(label == 0) & (Plabel == 1)]
    X_UNtoP = feature[(label == -1) & (Plabel == 1)]
    border20 = feature[(label != 1) & (Plabel == 1)][:, 0].min()
    # Uのうち、疑似ラベルとして選択されるもの(上位 5% or 2.5%)
    sn = int(sr * np.sum(label != 1))

    lab_pre_X_U = np.concatenate([label_U[:, np.newaxis], pred_U[:, np.newaxis], X_U], 1)
    high_index = np.argpartition(pred_U, -sn)[-sn:]
    lab_pre_X_U_high = lab_pre_X_U[high_index]
    X_UNtoPN  = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] == -1][:, -2:]
    X_UPtoPN  = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] ==  0][:, -2:]
    border5 = lab_pre_X_U_high[lab_pre_X_U_high[:, 0] != 1][:, -2].min()

    # 特徴量を散布図にする
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    # # 5%, 20%の領域を塗りつぶし
    # featxmax, featymin, featymax = feature[:, 0].max()+0.05, feature[:, 1].min()-0.05, feature[:, 1].max()+0.05
    # fillx = [border20,border5, border5, border20]
    # filly = [featymin, featymin, featymax, featymax]
    # if push_direction == "P":
    #     plt.fill(fillx, filly, color="orange", alpha=0.2)
    # fillcolor = "red" if push_direction == "P" else "blue"
    # fillx = [border5, featxmax, featxmax, border5]
    # plt.fill(fillx, filly, color=fillcolor, alpha=0.2)

    # プロット
    plt.scatter(X_PP[:, 0], X_PP[:, 1], marker="*", zorder=3, alpha=0.8, label="Positive", color="red")
    # plt.scatter(X_UP[:, 0], X_UP[:, 1], marker="o", zorder=1, alpha=0.6, label="UP", color="purple")
    # plt.scatter(X_UN[:, 0], X_UN[:, 1], marker="^", zorder=1, alpha=0.4, label="UN", color="blue")
    plt.scatter(X_U[:, 0],  X_U[:, 1],  marker="o", zorder=1, alpha=0.6, label="Unlabeled", color="mediumslateblue")

    # plt.scatter(X_UPtoP[:, 0],  X_UPtoP[:, 1],  marker="o", zorder=2, alpha=0.6, color="coral")
    # plt.scatter(X_UNtoP[:, 0],  X_UNtoP[:, 1],  marker="^", zorder=2, alpha=0.6, color="teal")
    # plt.scatter(X_UPtoPN[:, 0], X_UPtoPN[:, 1], marker="o", zorder=2, alpha=0.6, color="olive")
    # plt.scatter(X_UNtoPN[:, 0], X_UNtoPN[:, 1], marker="^", zorder=2, alpha=0.6, color="orange")

    # 5%, 20%の境界を通る軸
    # left, right = plt.xlim()
    # up, low = plt.ylim()
    # if push_direction == "P":
    #     plt.plot(np.full(feature.shape[0], border20), feature[:, 1], color="orange", alpha=0.5)
    # plt.plot(np.full(feature.shape[0], border5),  feature[:, 1], color=fillcolor, alpha=0.5)
    # plt.xlim(left, right)
    # plt.ylim(up, low)

    plt.axis("off")
    # dir = "Top" if push_direction == "P" else "Bottom"
    # ax.set_xlabel(f"Ranking by P-classification({dir}-push)")
    # ax.set_ylabel("First principal component", size=20)
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.5)

    save_path_vector = os.path.join(savepath, f"check_PCA_PC_{push_direction}-ch{check_num:02}-lap{lap:02}2.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0.1)
    # plt.show()

    plt.close()

def main(root_path, check_num, lap, push_direction, PU_num=1, PC_num=1):
    matplotlib_init()
    paths = make_paths(root_path, check_num, lap, push_direction, PU_num, PC_num)
    ## path list ##
    # 0: feature
    # 1: label
    # 2: Plabel
    # 3: predict
    # 4: grad
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    # いろいろ読み込み
    X = np.loadtxt(paths[0])
    Xmin = X.min()
    Xmax = X.max()
    X = (X - Xmin) / (Xmax - Xmin)
    # X = X if PN = "P" else -X

    label = np.loadtxt(paths[1])[:, 3]
    Plabel = np.loadtxt(paths[2])[:, 3]

    pred = np.loadtxt(paths[3])
    grad = np.loadtxt(paths[4])

    # PCA
    decomp = PCA(n_components=2)
    decomp.fit(X)

    X2 = np.transpose(X)
    X_decomp = np.array([grad.dot(X2), decomp.components_[0].dot(X2)]).transpose()
    grad = np.array([grad.dot(grad), decomp.components_[0].dot(grad)])

    # 分散比の計算
    evs = explained_variance_score(X, decomp.inverse_transform(X_decomp))

    # 可視化
    visualize2(check_num, lap, X_decomp, pred, label, Plabel, grad, evs, push_direction, paths[-1])
    # visualize3(check_num, lap, X_decomp, pred, label, Plabel, grad, evs, push_direction, paths[-1])


if __name__ == "__main__":
    main(root_path, check_num, lap, push_direction)
    print("finished")