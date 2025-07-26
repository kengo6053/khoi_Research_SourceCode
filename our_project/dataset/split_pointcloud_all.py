import os
import argparse
import pandas as pd
import numpy as np
import json

def split_excel_to_files(input_file, output_dir, output_format='txt', prefix='row', n=3):
    """
    Excel, CSV, またはテキストファイルを読み込み、各行を個別のファイルとして保存します。
    さらに、1行をn個ごとに分割して改行し、最終的に.npy形式で保存します。

    Args:
        input_file (str): 入力のExcel, CSV, またはテキストファイルのパス。
        output_dir (str): 出力ファイルを保存するディレクトリ。
        output_format (str): 中間出力ファイルの形式（'txt', 'csv', 'json'）。
        prefix (str): 出力ファイル名のプレフィックス。
        n (int): 行ごとに分割するデータ数。デフォルトは3。
    """
    # 入力ファイルの存在確認
    if not os.path.isfile(input_file):
        print(f"Error: 入力ファイル '{input_file}' が存在しません。")
        return

    # 出力ディレクトリの作成（存在しない場合）
    os.makedirs(output_dir, exist_ok=True)

    # ファイル拡張子に基づいて読み込み方法を選択
    _, file_extension = os.path.splitext(input_file)
    file_extension = file_extension.lower()

    try:
        if file_extension in ['.xlsx', '.xls']:
            # Excelファイルの読み込み
            df = pd.read_excel(input_file)
        elif file_extension == '.csv':
            # CSVファイルの読み込み
            df = pd.read_csv(input_file, header=None)
        elif file_extension == '.txt':
            # テキストファイルの読み込み
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame([line.strip().split() for line in lines if line.strip()])
        else:
            print(f"Error: 未対応のファイル形式 '{file_extension}' です。")
            print("Excel (.xlsx, .xls), CSV (.csv), またはテキスト (.txt) を使用してください。")
            return
    except Exception as e:
        print(f"Error: ファイルの読み込みに失敗しました。詳細: {e}")
        return

    # 行数を出力
    total_rows = len(df)
    print(f"Total number of rows in the input file: {total_rows}")

    # 最初の行のshapeを出力（参考表示）
    if not df.empty:
        first_row_shape = len(df.iloc[0])
        print(f"Shape of the first row: {first_row_shape}")

    # 各行を個別のファイルとして保存
    for idx, row in enumerate(df.itertuples(index=False, name=None)):
        if not any(row):  # 空行をスキップ
            print(f"Skipping empty row at index {idx}")
            continue

        # 行データをfloat型リストとして取得
        try:
            row_values = [float(value) for value in row]
        except ValueError as ve:
            print(f"Warning: 行 {idx} の変換に失敗しました: {ve}")
            continue

        # n個ごとに分割
        split_data = [row_values[i:i + n] for i in range(0, len(row_values), n)]

        # .npy ファイル名を生成 (例: 0000.npy, 0001.npy, ...)
        npy_file_name = f"{idx:04d}.npy"
        npy_file_path = os.path.join(output_dir, npy_file_name)

        # 分割後のデータを .npy ファイルとして保存
        try:
            np.save(npy_file_path, split_data)
            print(f"Saved .npy file: {npy_file_path}")
        except Exception as e:
            print(f"Error: .npy ファイル '{npy_file_path}' の保存に失敗しました。詳細: {e}")

def parse_arguments():
    """
    コマンドライン引数を解析します。

    Returns:
        argparse.Namespace: 解析された引数。
    """
    parser = argparse.ArgumentParser(
        description='Split an Excel, CSV, or text file into multiple files, one per row.'
    )
    parser.add_argument(
        '--parent_dir',
        type=str,
        default='${WORK_DIR}/our_project/dataset',
        help='複数のPMフォルダが入っている親ディレクトリのパス (例: /path/to/dataset).'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='csv',
        choices=['txt', 'csv', 'json'],
        help='中間出力ファイルの形式。デフォルトは csv。'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='row',
        help='出力ファイル名のプレフィックス。デフォルトは "row"。'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=3,
        help='1行内をn個ごとに分割。デフォルトは3。'
    )
    return parser.parse_args()

def main():
    # コマンドライン引数の解析
    args = parse_arguments()

    parent_dir = args.parent_dir

    # parent_dir にあるフォルダをすべて取得 (ファイルは除外・隠しフォルダ等は除外する想定)
    subfolders = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and not d.startswith('.')
    ]

    print(f"Parent directory: {parent_dir}")
    print(f"Detected folders: {subfolders}")

    # 各サブフォルダをループし、その中にある 3DPoints.csv を処理
    for folder_name in subfolders:
        input_file = os.path.join(parent_dir, folder_name, '3DPoints.csv')
        output_dir = os.path.join(parent_dir, folder_name, 'point_cloud')

        # 3DPoints.csv がない場合はスキップ（必要に応じてエラーでも可）
        if not os.path.isfile(input_file):
            print(f"Warning: {folder_name} 内に 3DPoints.csv がありません。スキップします。")
            continue

        print(f"\n--- Processing folder: {folder_name} ---")
        print(f"Input file: {input_file}")
        print(f"Output dir: {output_dir}")

        # 分割して .npy 出力
        split_excel_to_files(
            input_file=input_file,
            output_dir=output_dir,
            output_format=args.format,
            prefix=args.prefix,
            n=args.n
        )

if __name__ == "__main__":
    main()
