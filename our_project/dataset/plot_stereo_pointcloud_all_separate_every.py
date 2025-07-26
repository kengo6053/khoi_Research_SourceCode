import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# test_dataset のパスを指定
test_dataset_dir = "${WORK_DIR}/our_project/dataset/test_dataset"

# 指定された範囲
x_range = (32, -32)
y_range = (0, 64)   # y座標を逆順に設定
z_range = (-3, 6)   # z座標の範囲

# カラーマップの定義
cmap = plt.cm.jet  # 青から黄色へのグラデーション

def save_plot(x, y, z, x_range, y_range, z_range, file_name, view_name, output_folder,
              elev, azim, reverse_x=False, swap_xy=False):
    """
    3Dのscatterプロットを作成して画像として保存する関数。
    view_folder (output_folder/view_name) 内に画像を保存する。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(vmin=z.min(), vmax=z.max())
    
    # x と y を入れ替える場合 (俯瞰)
    if swap_xy:
        sc = ax.scatter(y, x, z, c=z, cmap=cmap, s=1, norm=norm)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(y_range[::-1])  # y の範囲を逆順
        ax.set_ylim(x_range)
    else:
        sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=1, norm=norm)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        if reverse_x:
            ax.set_xlim(x_range[::-1])  # x軸を逆順に設定 (zx 面のみ)
        else:
            ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    ax.set_zlabel("z")
    ax.set_zlim(z_range)
    
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("z")
    ax.view_init(elev=elev, azim=azim)
    
    # view_name 用のサブフォルダを作成して、その中に画像を保存
    view_folder = os.path.join(output_folder, view_name)
    os.makedirs(view_folder, exist_ok=True)

    output_file_path = os.path.join(view_folder, f"{file_name}_{view_name}_view.png")
    plt.savefig(output_file_path)
    print(f"プロット結果 ({view_name}方向) を保存しました: {output_file_path}")
    plt.close(fig)

# test_dataset_dir 内のすべてのフォルダを取得 (例: EX00PM01, EX01PM06, EX01PM15, ...)
parent_dirs = [
    d for d in os.listdir(test_dataset_dir)
    if os.path.isdir(os.path.join(test_dataset_dir, d))
]

for parent_dir_name in parent_dirs:
    parent_dir_path = os.path.join(test_dataset_dir, parent_dir_name)

    # point_cloud フォルダを入力、plot_pointcloud フォルダを出力に設定
    input_folder = os.path.join(parent_dir_path, "point_cloud")
    output_folder = os.path.join(parent_dir_path, "plot_pointcloud")

    # point_cloud フォルダが存在しない場合はスキップ
    if not os.path.isdir(input_folder):
        print(f"Skipping {parent_dir_name}: 'point_cloud' folder not found.")
        continue

    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n=== 処理中のディレクトリ: {parent_dir_path} ===")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")

    # 入力フォルダ内のすべての npy ファイルを再帰的に取得
    npy_files = glob.glob(os.path.join(input_folder, "**/*.npy"), recursive=True)

    if not npy_files:
        print(f"No .npy files found in {input_folder}. Skipping...")
        continue

    # すべての npy ファイルに対して処理を実行
    for npy_file_path in npy_files:
        print(f"  -> 読み込み中: {npy_file_path}")
        # ファイル名を取得
        file_name = os.path.splitext(os.path.basename(npy_file_path))[0]
        
        # npyファイルの読み込み
        data = np.load(npy_file_path, allow_pickle=True)
        
        # shape (N, 4) を想定: (y, x, z, intensity) など
        point_cloud_data = data

        # x, y, z を取得 (想定: data[:, 0]=y, data[:, 1]=x, data[:, 2]=z)
        x = point_cloud_data[:, 1]  # x座標
        y = point_cloud_data[:, 0]  # y座標
        z = point_cloud_data[:, 2]  # z座標

        # 指定された範囲にフィルタリング
        filtered_indices = (
            (x >= x_range[1]) & (x <= x_range[0]) &
            (y >= y_range[0]) & (y <= y_range[1]) &
            (z >= z_range[0]) & (z <= z_range[1])
        )
        x = x[filtered_indices]
        y = y[filtered_indices]
        z = z[filtered_indices]

        if x.size == 0:
            print(f"    * {file_name}.npy: No points in the specified range. Skipping plotting.")
            continue

        # 各方向のプロットを保存
        save_plot(x, y, z, x_range, y_range, z_range,
                  file_name, "xy",  output_folder, elev=90,  azim=-90)              # xy面
        save_plot(x, y, z, x_range, y_range, z_range,
                  file_name, "yz",  output_folder, elev=0,   azim=0)                # yz面
        save_plot(x, y, z, x_range, y_range, z_range,
                  file_name, "zx",  output_folder, elev=0,   azim=90, reverse_x=True)  # zx面
        save_plot(x, y, z, x_range, y_range, z_range,
                  file_name, "top", output_folder, elev=30,  azim=45, swap_xy=True)    # 俯瞰

print("\nすべての処理が完了しました。")
