import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 3D点群データの読み込み
npy_file_path = "${WORK_DIR}/data/coke_dataset_23_11/Routes_Scenario3_Town01_curved_Seed1000/Scenario3_Town01_curved_route1_11_23_20_23_26/lidar/0000.npy"  # npyファイルのパスを指定
output_folder = "${WORK_DIR}/our_project/dataset/view_data"     # 保存するフォルダを指定

# npyファイルの読み込み
data = np.load(npy_file_path, allow_pickle=True)

# 点群データを抽出
point_cloud_data = data[1]  # shape: (22100, 4)

# x, y, z, intensity を取得
x = point_cloud_data[:, 0]  # x座標
y = point_cloud_data[:, 1]  # y座標
z = point_cloud_data[:, 2]  # z座標

# 指定された範囲にフィルタリング
x_range = (16, -16)
y_range = (0, -32)  # y座標を逆順に設定
filtered_indices = (x >= x_range[1]) & (x <= x_range[0]) & (y >= y_range[1]) & (y <= y_range[0])
x = x[filtered_indices]
y = y[filtered_indices]
z = z[filtered_indices]

# カラーマップの手動範囲設定（高さに基づく）
vmin = z.min()  # zの最小値
vmax = z.max()  # zの最大値

# 3Dデータのプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# カラーマップを高さ（z）に基づいて設定
cmap = plt.cm.jet  # 青から黄色へのグラデーション
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# プロット
sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=1, norm=norm)  # 高さで色分け
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# カラーバーを追加
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("z")
cbar.set_ticks(np.linspace(vmin, vmax, num=6))  # 高さの範囲で適切に目盛りを設定

# 軸の範囲を設定
ax.set_xlim(x_range)
ax.set_ylim(y_range)  # y_range を適用（逆順）
ax.set_zlim([z.min(), z.max()])

# 背景色を白に変更
fig.patch.set_facecolor('white')

# 保存フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# プロット結果を保存
output_file_path = os.path.join(output_folder, "3d_scatter_plot_y_reversed.png")
plt.savefig(output_file_path)
print(f"プロット結果を保存しました: {output_file_path}")

# プロットを表示
plt.show()
