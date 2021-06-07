import sys
import os

from laplacian import laplacian

args = sys.argv

# args: 1:filter_name, 2: image_path ,3: outputDirPath
filter_name = args[1]
image_path = args[2]
output_dir_path = args[3]

# outputフォルダの作成
if not os.path.exists(output_dir_path):  # ディレクトリがなかったら
    os.makedirs(output_dir_path)

if filter_name == "laplacian":
    laplacian(image_path, output_dir_path)
else:
    raise ValueError("filter not found")
