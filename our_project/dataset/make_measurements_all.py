import os
import csv
import json
import argparse

def convert_csv_to_json(input_file, output_folder):
    """
    CSVファイル (VehicleMove.csv) を行ごとに読み込み、
    velocity, acceleration の情報を JSONファイル として出力する。

    Args:
        input_file (str): CSVファイルのパス
        output_folder (str): JSONファイルを出力する先のフォルダ
    """
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # CSVファイルを1行ずつ読み込み、JSONとして出力
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for index, row in enumerate(csv_reader):
            # 1つ目と2つ目の値を取り出す
            velocity = float(row[0])
            acceleration = float(row[1])

            # 辞書形式でデータを作成
            data = {
                "velocity": velocity,
                "acceleration": acceleration
            }

            # ファイル名を作成 (例: 0000.json, 0001.json, ...)
            file_name = f"{index:04d}.json"
            file_path = os.path.join(output_folder, file_name)

            # JSONファイルに書き込む
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)

    print(f"JSONファイルがフォルダ '{output_folder}' に出力されました。")

def parse_arguments():
    """
    コマンドライン引数を解析する。
    """
    parser = argparse.ArgumentParser(
        description='Convert VehicleMove.csv in each PM folder to JSON files.'
    )
    parser.add_argument(
        '--parent_dir',
        type=str,
        default='${WORK_DIR}/our_project/dataset/train_dataset/train',
        help='PMフォルダが入っている親ディレクトリのパス (例: /path/to/dataset/train)'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    parent_dir = args.parent_dir

    # 親ディレクトリ内にあるフォルダをすべて取得 (隠しフォルダ等は除外する想定)
    subfolders = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')
    ]

    print(f"Parent directory: {parent_dir}")
    print(f"Detected folders: {subfolders}")

    # 各サブフォルダに対して VehicleMove.csv を探し、JSON化
    for folder_name in subfolders:
        input_file = os.path.join(parent_dir, folder_name, 'VehicleMove.csv')
        output_folder = os.path.join(parent_dir, folder_name, 'measurements')

        if not os.path.isfile(input_file):
            print(f"Warning: {folder_name} 内に VehicleMove.csv がありません。スキップします。")
            continue

        print(f"\n--- Processing folder: {folder_name} ---")
        print(f"Input file: {input_file}")
        print(f"Output folder: {output_folder}")

        # CSV -> JSON変換
        convert_csv_to_json(input_file, output_folder)

if __name__ == '__main__':
    main()
