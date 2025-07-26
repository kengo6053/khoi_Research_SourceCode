import os

def rename_folder(parent_dir, old_name, new_name):
    """
    親フォルダ内の特定の名前のフォルダを新しい名前に変更する。

    Args:
        parent_dir (str): 親フォルダのパス
        old_name (str): 変更前のフォルダ名
        new_name (str): 変更後のフォルダ名
    """
    # 親フォルダ内のサブフォルダをすべて取得
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        if os.path.isdir(folder_path):
            # 「クリア画像(右)」というフォルダがあるか確認
            target_folder = os.path.join(folder_path, old_name)
            if os.path.isdir(target_folder):
                new_folder_path = os.path.join(folder_path, new_name)
                try:
                    os.rename(target_folder, new_folder_path)
                    print(f"Renamed: {target_folder} -> {new_folder_path}")
                except Exception as e:
                    print(f"Failed to rename {target_folder} to {new_folder_path}: {e}")
            else:
                print(f"No folder named '{old_name}' in {folder_path}")

if __name__ == "__main__":
    # 親フォルダのパス
    parent_directory = "${WORK_DIR}/our_project/dataset/new_dataset"

    # フォルダ名の変更
    old_folder_name = "クリア画像(右)"
    new_folder_name = "gray_right"

    rename_folder(parent_directory, old_folder_name, new_folder_name)
