import numpy as np
import os
import argparse

def transform_point_cloud(data):
    """
    入力の点群データ（(N, ?)-shapeのnumpy配列）について、
    各行の1要素目と2要素目を入れ替えた上で、入れ替え後の2要素目に -1 を掛けた結果を返す。
    """
    transformed = data.copy()
    # 1要素目と2要素目の入れ替え
    transformed[:, [0, 1]] = transformed[:, [1, 0]]
    # 入れ替え後の2要素目に -1 を掛ける
    transformed[:, 1] = transformed[:, 1] * -1
    return transformed

def process_directory(base_dir):
    """
    指定したベースディレクトリ内の各サブディレクトリ（例：EX00AM00, EX00AM02, …）について、
    その中の「point_cloud」フォルダ内のすべての .npy ファイルを処理し、
    処理結果を同じサブディレクトリ内の「point_cloud_transformed」フォルダに保存する。
    """
    # ベースディレクトリ内のすべてのサブディレクトリをループ
    for sub_dir in os.listdir(base_dir):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        # point_cloud フォルダのパスを作成
        point_cloud_dir = os.path.join(sub_dir_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            print(f"Skipping {sub_dir_path}: 'point_cloud' directory not found.")
            continue

        # 出力先の point_cloud_transformed フォルダのパスを作成（存在しなければ作成）
        transformed_dir = os.path.join(sub_dir_path, "point_cloud_transformed")
        os.makedirs(transformed_dir, exist_ok=True)

        # point_cloud ディレクトリ内の .npy ファイルすべてを処理
        for fname in os.listdir(point_cloud_dir):
            if fname.lower().endswith(".npy"):
                input_path = os.path.join(point_cloud_dir, fname)
                output_path = os.path.join(transformed_dir, fname)
                try:
                    data = np.load(input_path)
                    transformed_data = transform_point_cloud(data)
                    np.save(output_path, transformed_data)
                    print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing file {input_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="各サブディレクトリ内の 'point_cloud' フォルダにある npy ファイルを変換し、同じサブディレクトリ内に 'point_cloud_transformed' フォルダを作成して保存します。"
    )
    parser.add_argument("--base_dir", type=str, required=True, help="サブディレクトリ（例: EX00AM00, EX00AM02, ...）が含まれるベースディレクトリのパス")
    args = parser.parse_args()
    process_directory(args.base_dir)
