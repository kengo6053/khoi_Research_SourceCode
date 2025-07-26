import os
import argparse
import pandas as pd
import numpy as np
import json

def split_excel_to_files(input_file, output_dir, output_format='txt', prefix='row', n=3):
    """
    ExcelまたはCSVファイルを読み込み、各行を個別のファイルとして保存します。
    さらに、1行をn個ごとに分割して改行し、最終的に.npy形式で保存します。

    Args:
        input_file (str): 入力のExcelまたはCSVファイルのパス。
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
            df = pd.read_csv(input_file)
        else:
            print(f"Error: 未対応のファイル形式 '{file_extension}' です。Excel (.xlsx, .xls) または CSV (.csv) を使用してください。")
            return
    except Exception as e:
        print(f"Error: ファイルの読み込みに失敗しました。詳細: {e}")
        return

    # 行数の出力
    num_rows = len(df)
    print(f"入力ファイルの行数: {num_rows}")
    
def parse_arguments():
    """
    コマンドライン引数を解析します。

    Returns:
        argparse.Namespace: 解析された引数。
    """
    parser = argparse.ArgumentParser(description='Split an Excel or CSV file into multiple files, one per row.')
    parser.add_argument('input_file', type=str, help='Path to the input Excel or CSV file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output files.')
    parser.add_argument('--format', type=str, default='csv', choices=['txt', 'csv', 'json'],
                        help='Intermediate output file format. Options: txt, csv, json. Default is txt.')
    parser.add_argument('--prefix', type=str, default='row', help='Prefix for output file names. Default is "row".')
    parser.add_argument('--n', type=int, default=3, help='Number of elements per line in the final output. Default is 3.')

    return parser.parse_args()

def main():
    # コマンドライン引数の解析
    args = parse_arguments()

    # 関数の呼び出し
    split_excel_to_files(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_format=args.format,
        prefix=args.prefix,
        n=args.n
    )

if __name__ == "__main__":
    main()

