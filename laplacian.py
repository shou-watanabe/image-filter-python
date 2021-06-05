# 線形空間フィルタリングを行う
import cv2
import numpy as np
import sys
import os


def laplacian(image_path, output_dir_path):
    print('ラプラシアンフィルタリングを行う')

    # 画像データ読み込み（原画像のカラー状態を保持して読み込む）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img.ndim != 2:
        print('濃淡画像ではありません')
        sys.exit()

    # 画像の縦横サイズの取得
    h, w = img.shape[:2]

    # サイズ，カラー，画素のデータタイプ
    print('入力画像ファイル = ', image_path)
    print('高さ = ', h, ',幅 = ', w)

    # 係数行列の定義

    # 四近傍ラプラシアン
    kernel1 = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], np.float32)

    # 八近傍ラプラシアン
    kernel2 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]], np.float32)

    # 画素データの型変換（filter2D関数に渡す前に型変換しておく、重要）
    img = img.astype(np.float32)

    # カーネルでフィルタリング
    out4 = cv2.filter2D(img, -1, kernel1)  # -1:入力画像（img）と同じデータ型でreturnする
    out8 = cv2.filter2D(img, -1, kernel2)

    outimg4 = cv2.convertScaleAbs(out4)
    outimg8 = cv2.convertScaleAbs(out8)

    # 出力する画像ファイル名を作る
    basename = os.path.basename(image_path)[0:image_path.rfind('.')]
    wfname4 = output_dir_path + "/" + basename + '_laplacian4.png'
    wfname8 = output_dir_path + "/" + basename + '_laplacian8.png'

    # 出力
    cv2.imwrite(wfname4, outimg4)
    cv2.imwrite(wfname8, outimg8)
