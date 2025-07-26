import os
import shutil

def delete_pointcloud_items(parent_dir, target_name="pointcloud"):
    """
    親フォルダ内のすべてのサブフォルダを探索し、
    指定された名前のフォルダやファイルを削除する。

    Args:
        parent_dir (str): 親フォルダのパス
        target_name (str): 削除対象の名前 (フォルダまたはファイル)
    """
    # 親フォルダ内のサブフォルダを探索
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        if os.path.isdir(folder_path):
            # 削除対象 (フォルダやファイル) を確認
            target_item = os.path.join(folder_path, target_name)
            if os.path.exists(target_item):  # フォルダまたはファイルが存在する場合
                try:
                    if os.path.isdir(target_item):  # フォルダの場合
                        shutil.rmtree(target_item)  # フォルダを再帰的に削除
                        print(f"Deleted folder: {target_item}")
                    else:  # ファイルの場合
                        os.remove(target_item)  # ファイルを削除
                        print(f"Deleted file: {target_item}")
                except Exception as e:
                    print(f"Failed to delete {target_item}: {e}")
            else:
                print(f"No item named '{target_name}' in {folder_path}")

if __name__ == "__main__":
    # 親フォルダのパス
    parent_directory = "${WORK_DIR}/our_project/dataset/train_dataset/train"

    # 削除対象の名前 (フォルダまたはファイル)
    target_name = "VehicleMove.csv"

    # 削除処理を実行
    delete_pointcloud_items(parent_directory, target_name)
