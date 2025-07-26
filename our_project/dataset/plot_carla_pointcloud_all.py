import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# 入力ディレクトリと出力ディレクトリを指定
input_folder = "${WORK_DIR}/our_project/dataset/view_data/pointcloud_test"  # npyファイルが格納されているフォルダ
output_folder = "${WORK_DIR}/our_project/dataset/view_data/plot_pointcloud_test2"  # プロット結果を保存するフォルダ

# 指定された範囲
x_range = (16, -16)
y_range = (0, -32)  # y座標を逆順に設定
z_range = (-3, 6)  # z座標の範囲

# 保存フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# カラーマップの定義
cmap = plt.cm.jet  # 青から黄色へのグラデーション

# 関数: 各方向のプロットを保存
def save_plot(x, y, z, x_range, y_range, z_range, file_name, view_name, elev, azim, reverse_x=False, swap_xy=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # カラーマップの正規化を z_range に合わせる
    norm = plt.Normalize(vmin=z_range[0], vmax=z_range[1])  # z_range を反映
    
    # x と y を入れ替える場合 (俯瞰)
    if swap_xy:
        sc = ax.scatter(y, x, z, c=z, cmap=cmap, s=1, norm=norm)  # x と y を入れ替えてプロット
        ax.set_xlabel("y")  # ラベルを入れ替え
        ax.set_ylabel("x")  # ラベルを入れ替え
        ax.set_xlim(y_range[::-1])  # y の範囲を逆順
        ax.set_ylim(x_range)  # x の範囲をそのまま
    else:
        sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=1, norm=norm)  # 通常プロット
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if reverse_x:
            ax.set_xlim(x_range[::-1])  # x軸を逆順に設定 (zx 面のみ)
        else:
            ax.set_xlim(x_range)  # x軸を通常設定
        ax.set_ylim(y_range)

    ax.set_zlabel("z")
    ax.set_zlim(z_range)
    
    # カラーバーの設定
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("z")
    cbar.set_ticks(np.linspace(z_range[0], z_range[1], num=6))  # z_range に基づいて目盛りを設定

    ax.view_init(elev=elev, azim=azim)
    output_file_path = os.path.join(output_folder, f"{file_name}_{view_name}_view.png")
    plt.savefig(output_file_path)
    print(f"プロット結果 ({view_name}方向) を保存しました: {output_file_path}")
    plt.close(fig)

# 入力フォルダ内のすべてのnpyファイルを取得
npy_files = glob.glob(os.path.join(input_folder, "**/*.npy"), recursive=True)

# すべてのnpyファイルに対して処理を実行
for npy_file_path in npy_files:
    print(f"処理中: {npy_file_path}")
    # ファイル名を取得
    file_name = os.path.splitext(os.path.basename(npy_file_path))[0]
    
    # npyファイルの読み込み
    data = np.load(npy_file_path, allow_pickle=True)
    point_cloud_data = data
    #[1]# shape: (N, 4)

    # x, y, z, intensity を取得
    x = point_cloud_data[:, 0]  # x座標
    y = point_cloud_data[:, 1]  # y座標
    z = point_cloud_data[:, 2]  # z座標

    # 指定された範囲にフィルタリング
    filtered_indices = (
        (x >= x_range[1]) & (x <= x_range[0]) &
        (y >= y_range[1]) & (y <= y_range[0]) &
        (z >= z_range[0]) & (z <= z_range[1])
    )
    x = x[filtered_indices]
    y = y[filtered_indices]
    z = z[filtered_indices]

    # 各方向のプロットを保存
    save_plot(x, y, z, x_range, y_range, z_range, file_name, "xy", elev=90, azim=-90)  # xy面
    save_plot(x, y, z, x_range, y_range, z_range, file_name, "yz", elev=0, azim=0)    # yz面
    save_plot(x, y, z, x_range, y_range, z_range, file_name, "zx", elev=0, azim=90, reverse_x=True)  # zx面 (x軸を逆に)
    save_plot(x, y, z, x_range, y_range, z_range, file_name, "top", elev=30, azim=45, swap_xy=True)  # 俯瞰 (x と y を入れ替え)
