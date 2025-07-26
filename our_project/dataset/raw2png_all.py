import os
import argparse
import numpy as np
from PIL import Image

def convert_raw_to_png(parent_folder, width, height):
    """
    親フォルダ(parent_folder)以下のgrayフォルダ内にあるRAWファイルをPNGに変換し、
    同フォルダ階層のpng_folderに出力する。
    """
    # grayフォルダへのパス
    input_folder = os.path.join(parent_folder, "gray_right")
    # png_folderへのパス
    output_folder = os.path.join(parent_folder, "gray_right_png_folder")
    
    # 出力フォルダを作成（存在しない場合のみ）
    os.makedirs(output_folder, exist_ok=True)

    # grayフォルダ内のRAWファイル一覧を取得
    if not os.path.exists(input_folder):
        print(f"grayフォルダが存在しません: {input_folder}")
        return

    files = [f for f in os.listdir(input_folder) if f.endswith(".raw")]

    if not files:
        print(f"grayフォルダ内にRAWファイルが見つかりません: {input_folder}")
        return

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace(".raw", ".png"))

        try:
            # RAWファイルを読み込む
            with open(input_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)

            # RAWデータを2Dアレイに変換
            if raw_data.size != width * height:
                print(f"Error: {file_name} のサイズが指定した幅・高さ({width}x{height})と一致しません。")
                continue

            image_array = raw_data.reshape((height, width))

            # 画像として保存
            image = Image.fromarray(image_array)
            image.save(output_path)

            print(f"Converted: {input_path} -> {output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


def process_all_subdirectories(root_folder, width, height):
    """
    ルートフォルダ以下のすべてのサブディレクトリについてRAWからPNGへの変換を行う
    """
    # ルートフォルダ内のすべてのサブディレクトリを取得
    subdirectories = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    for subdir in subdirectories:
        print(f"Processing directory: {subdir}")
        convert_raw_to_png(subdir, width, height)


if __name__ == "__main__":
    # コマンドライン引数でルートフォルダ、幅、高さを受け取る
    parser = argparse.ArgumentParser(description='Convert RAW images in "gray" folders to PNGs in "png_folder" for all subdirectories.')
    parser.add_argument('--root_folder', required=True, help='Path to the root folder containing subdirectories (e.g. test_dataset).')
    parser.add_argument('--width', type=int, default=1492, help='Image width (default: 640).')
    parser.add_argument('--height', type=int, default=412, help='Image height (default: 480).')
    args = parser.parse_args()

    process_all_subdirectories(args.root_folder, args.width, args.height)
