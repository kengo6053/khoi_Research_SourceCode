import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 3D点群データの読み込み
npy_file_path = "${WORK_DIR}/our_project/dataset/view_data/EX00AM03/point_cloud/0000.npy"  # npyファイルのパスを指定
output_folder = "${WORK_DIR}/our_project/dataset/view_data"     # 保存するフォルダを指定

# npyファイルの読み込み
data = np.load(npy_file_path, allow_pickle=True)

# 点群データを抽出
point_cloud_data = data  # shape: (22100, 4)

# x, y, z, intensity を取得
x = point_cloud_data[:, 1]  # x座標
y = point_cloud_data[:, 0]  # y座標
z = point_cloud_data[:, 2]  # z座標

# 指定された範囲にフィルタリング
x_range = (16, -16)
y_range = (0, 32)  # y座標を逆順に設定
filtered_indices = (x >= x_range[1]) & (x <= x_range[0]) & (y >= y_range[0]) & (y <= y_range[1])
x = x[filtered_indices]
y = y[filtered_indices]
z = z[filtered_indices]

# カラーマップの手動範囲設定（高さに基づく）
vmin = z.min()  # zの最小値
vmax = z.max()  # zの最大値

# カラーマップを定義
cmap = plt.cm.jet  # 青から黄色へのグラデーション
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# 3Dデータのプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# カラーマップを高さ（z）に基づいて設定
sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=1, norm=norm)  # 高さで色分け
ax.set_xlabel("y")
ax.set_ylabel("x")
ax.set_zlabel("z")

# カラーバーを追加
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("z")

# 軸の範囲を設定
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_zlim([z.min(), z.max()])

# **xy方向**の視点に設定
ax.view_init(elev=90, azim=-90)  # 上から見た視点

# 保存フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# プロット結果を保存
output_file_path_xy = os.path.join(output_folder, "3d_scatter_plot_stereo_xy_view.png")
plt.savefig(output_file_path_xy)
print(f"プロット結果 (xy方向) を保存しました: {output_file_path_xy}")

plt.show()
