import os
import csv
import json
import argparse

def convert_csvs_to_json(vehicle_file, basic_file, output_folder):
    """
    VehicleMove.csv と basic_accelaration.csv を読み込み、
    velocity, acceleration, basic_acceleration の情報を JSON ファイルとして出力する。

    Args:
        vehicle_file (str): VehicleMove.csv のパス
        basic_file (str): basic_accelaration.csv のパス
        output_folder (str): JSON ファイルを出力する先のフォルダ
    """
    os.makedirs(output_folder, exist_ok=True)

    # basic_accelaration.csv の1列目をリスト化
    basic_vals = []
    if os.path.isfile(basic_file):
        with open(basic_file, 'r', encoding='utf-8') as bf:
            basic_reader = csv.reader(bf)
            for row in basic_reader:
                try:
                    basic_vals.append(float(row[0]))
                except (IndexError, ValueError):
                    basic_vals.append(None)
    else:
        print(f"Warning: basic file not found: {basic_file}")

    # VehicleMove.csv を読み込み、JSON に変換
    with open(vehicle_file, 'r', encoding='utf-8') as vf:
        vehicle_reader = csv.reader(vf)
        for idx, row in enumerate(vehicle_reader):
            try:
                velocity = float(row[0])
                acceleration = float(row[1])
            except (IndexError, ValueError):
                print(f"Skipping invalid row {idx} in {vehicle_file}: {row}")
                continue

            basic_acc = basic_vals[idx] if idx < len(basic_vals) else None

            data = {
                "velocity": velocity,
                "acceleration": acceleration,
                "basic_acceleration": basic_acc
            }

            file_name = f"{idx:04d}.json"
            out_path = os.path.join(output_folder, file_name)
            with open(out_path, 'w', encoding='utf-8') as jf:
                json.dump(data, jf, indent=4)

    print(f"Exported JSON files to '{output_folder}'")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert VehicleMove.csv and basic_accelaration.csv in each PM folder to JSON files.'
    )
    parser.add_argument(
        '--parent_dir',
        type=str,
        default='${WORK_DIR}/our_project/dataset/train_dataset/train',
        help='PM フォルダが入っている親ディレクトリのパス'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    parent_dir = args.parent_dir

    subfolders = [d for d in os.listdir(parent_dir)
                  if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')]

    print(f"Parent directory: {parent_dir}")
    print(f"Detected folders: {subfolders}")

    for folder in subfolders:
        vehicle_file = os.path.join(parent_dir, folder, 'VehicleMove.csv')
        basic_file = os.path.join(parent_dir, folder, 'basic_accelaration.csv')
        output_folder = os.path.join(parent_dir, folder, 'measurements')

        if not os.path.isfile(vehicle_file):
            print(f"Warning: {vehicle_file} が見つかりません。スキップします。")
            continue

        print(f"\n--- Processing folder: {folder} ---")
        print(f"Vehicle file: {vehicle_file}")
        print(f"Basic file:   {basic_file}")
        print(f"Output dir:   {output_folder}")

        convert_csvs_to_json(vehicle_file, basic_file, output_folder)

if __name__ == '__main__':
    main()
