import os
import argparse

def count_files_in_subfolders(root_folder, output_file):
    """
    指定したルートフォルダ内の各親フォルダの
    gray、measurements、point_cloud、point_cloud_transformedフォルダに含まれるファイル数を出力し、
    一致しているフォルダでは一致している値を出力し、
    一致していないフォルダはピックアップして表示する。
    また、結果を指定したファイルに保存する。
    """
    # 各親フォルダを処理
    subdirectories = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    # カウント対象のフォルダ名
    target_folders = ["gray_right", "measurements", "point_cloud", "point_cloud_transformed"]
    
    result = {}
    totals = {folder: 0 for folder in target_folders}  # 合計値の初期化

    total_folders = len(subdirectories)  # 親フォルダの総数
    consistent_folders = []  # ファイル数が一致している親フォルダを記録
    inconsistent_folders = []  # ファイル数が一致していない親フォルダを記録

    # 結果を保存するためのリスト
    output_lines = []

    output_lines.append(f"Found {total_folders} parent folders to process.\n")

    for index, subdir in enumerate(subdirectories, start=1):
        folder_name = os.path.basename(subdir)
        output_lines.append(f"Processing folder {index}/{total_folders}: {folder_name}")
        result[folder_name] = {}

        folder_counts = []
        for target in target_folders:
            target_path = os.path.join(subdir, target)

            if os.path.exists(target_path) and os.path.isdir(target_path):
                # 対象フォルダ内のファイル数をカウント
                file_count = sum(1 for entry in os.scandir(target_path) if entry.is_file())
                result[folder_name][target] = file_count
                folder_counts.append(file_count)
                totals[target] += file_count  # 合計値に加算
            else:
                # 対象フォルダが存在しない場合は0を記録
                result[folder_name][target] = 0
                folder_counts.append(0)

        # ファイル数が一致している場合
        if len(set(folder_counts)) == 1:
            consistent_folders.append((folder_name, folder_counts[0]))
        else:
            inconsistent_folders.append(folder_name)

    # 各親フォルダごとの結果を出力
    output_lines.append("\nSummary of file counts by folder:\n")
    for folder, counts in result.items():
        output_lines.append(f"Folder: {folder}")
        for target, count in counts.items():
            output_lines.append(f"  {target}: {count} files")
        output_lines.append("")  # 空行で区切る

    # 一致しているフォルダを出力
    consistent_total = sum(count for _, count in consistent_folders)  # 合計を計算
    output_lines.append("\nFolders with consistent file counts:\n")
    if consistent_folders:
        for folder, count in consistent_folders:
            output_lines.append(f"  {folder}: {count} files")
        output_lines.append(f"\nTotal files in consistent folders: {consistent_total}")  # 合計を出力
    else:
        output_lines.append("  None")

    # 一致していないフォルダを出力
    output_lines.append("\nFolders with inconsistent file counts:\n")
    if inconsistent_folders:
        for folder in inconsistent_folders:
            output_lines.append(f"  {folder}")
    else:
        output_lines.append("  None")

    # 結果をコンソールに出力
    for line in output_lines:
        print(line)

    # 結果をファイルに保存
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))
    print(f"\nResults have been written to {output_file}")

if __name__ == "__main__":
    # コマンドライン引数でルートフォルダと出力ファイルを指定
    parser = argparse.ArgumentParser(description='Count files in specific subfolders for each parent folder.')
    parser.add_argument('--root_folder', required=True, help='Path to the root folder containing parent folders.')
    parser.add_argument('--output_file', default="${WORK_DIR}/our_project/dataset/new_dataset2/result_file_num.txt", help='Path to the output file (default: output.txt).')
    args = parser.parse_args()

    count_files_in_subfolders(args.root_folder, args.output_file)
