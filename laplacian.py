# 線形空間フィルタリングを行う
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def laplacian(image_path, output_dir_path):
    # 画像データ読み込み（原画像のカラー状態を保持して読み込む）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 係数行列の定義
    # 四近傍ラプラシアン
    kernel4 = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], np.float32)

    # 八近傍ラプラシアン
    kernel8 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]], np.float32)

    # 画素データの型変換（filter2D関数に渡す前に型変換しておく、重要）
    img = img.astype(np.float32)

    # カーネルでフィルタリング
    out = cv2.filter2D(img, -1, kernel4)  # -1:入力画像（img）と同じデータ型でreturnする

    outimg = cv2.convertScaleAbs(out)

    (hist, bins) = np.histogram(img.flatten(), 256, [0, 256])

    (hhist, hbins) = np.histogram(out.flatten(), 256, [0, 256])

    fig = plt.figure()

    # グラフをプロット
    plt.plot(bins[:256], hist, label='before')
    plt.plot(hbins[:256], hhist, label='after')
    # グラフの凡例
    plt.legend()

    basename = os.path.basename(image_path)

    file_name = basename[0:basename.rfind('.')]

    wfname = os.path.join(output_dir_path, basename)

    # 出力
    cv2.imwrite(wfname, outimg)

    graph_file_name = os.path.join(output_dir_path, file_name + "_plt.png")

    fig.savefig(graph_file_name)
