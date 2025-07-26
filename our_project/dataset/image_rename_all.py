import os
import re
import argparse

def rename_files(target_directory):
    """
    与えられたディレクトリにあるファイルのうち、
    「数字_PIXRIGY_数字x数字.raw」という形式にマッチするファイルを集めて
    数字部分でソートした後、連番(0000, 0001, …)形式のファイル名にリネームする。
    """
    # 対象ファイル名を格納するリスト
    file_list = []

    # ファイル名を解析してリストに追加
    for file_name in os.listdir(target_directory):
        if re.match(r'^\d+_PIXRIGY_\d+x\d+\.raw$', file_name):
            # ファイル名から数字部分を抽出
            match = re.match(r'^(\d+)_PIXRIGY_\d+x\d+\.raw$', file_name)
            if match:
                number = int(match.group(1))
                file_list.append((number, file_name))

    # 数字部分でソート
    file_list.sort()

    # ファイル名を新しい形式に変更
    for index, (number, original_name) in enumerate(file_list):
        new_name = f"{index:04d}.raw"
        original_path = os.path.join(target_directory, original_name)
        new_path = os.path.join(target_directory, new_name)
        os.rename(original_path, new_path)
        print(f"Renamed: {original_name} -> {new_name}")

def parse_arguments():
    """
    コマンドライン引数を解析する。
    """
    parser = argparse.ArgumentParser(
        description='Rename RAW files in the gray directory of each PM folder.'
    )
    parser.add_argument(
        '--parent_dir',
        type=str,
        default='${WORK_DIR}/our_project/dataset/new_dataset',
        help='PMフォルダが入っている親ディレクトリのパス (例: /path/to/dataset/train)'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    parent_dir = args.parent_dir

    # 親ディレクトリ内にあるフォルダをすべて取得 (隠しフォルダ等は除外)
    subfolders = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')
    ]

    print(f"Parent directory: {parent_dir}")
    print(f"Detected folders: {subfolders}")

    # 各サブフォルダをループし、その中の gray ディレクトリを探す
    for folder_name in subfolders:
        gray_dir = os.path.join(parent_dir, folder_name, 'gray_right')

        if not os.path.isdir(gray_dir):
            print(f"Warning: {folder_name} フォルダに gray_right ディレクトリがありません。スキップします。")
            continue

        print(f"\n--- Processing folder: {folder_name} ---")
        print(f"Target directory: {gray_dir}")

        rename_files(gray_dir)

if __name__ == "__main__":
    main()
