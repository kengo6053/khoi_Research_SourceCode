import os
import re

def rename_files(target_directory):
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
        new_name = f"{index:04}.raw"
        original_path = os.path.join(target_directory, original_name)
        new_path = os.path.join(target_directory, new_name)
        os.rename(original_path, new_path)
        print(f"Renamed: {original_name} -> {new_name}")

if __name__ == "__main__":
    target_directory = '${WORK_DIR}/our_project/dataset/train_dataset/train_test/EX00PM06/gray'
    if os.path.isdir(target_directory):
        rename_files(target_directory)
    else:
        print("The specified directory does not exist.")
