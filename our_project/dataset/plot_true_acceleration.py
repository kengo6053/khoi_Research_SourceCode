import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import json
import argparse
from datetime import datetime

# データセットを定義
class JSONDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(
            [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.json')]
        )  # JSONファイルを名前順にソート
        if not self.files:
            raise FileNotFoundError(f"No JSON files found in directory: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            data = json.load(f)
        acceleration = data.get('acceleration', 0.0)
        return {
            'acceleration': torch.tensor(acceleration, dtype=torch.float32)
        }

# データローダを使って検証データを取得し、真値をプロット
def plot_true_acceleration(dataloader_val, save_folder=None):
    """
    検証データの加速度の真値をプロットする関数。

    Parameters:
    - dataloader_val: DataLoader
        検証用データの DataLoader。
    - save_folder: str
        グラフを保存するフォルダ（指定しない場合は表示のみ）。
    """
    true_accelerations = []

    # データローダから真値を収集
    for data in dataloader_val:
        accelerations = data['acceleration'].detach().cpu().numpy()
        true_accelerations.append(accelerations)

    # 真値を1次元に結合
    true_accelerations = np.concatenate(true_accelerations, axis=0)

    # グラフをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(true_accelerations, label='True Acceleration', color='blue')
    plt.title('True Acceleration Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration')
    plt.legend()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"true_acceleration_plot_{timestamp}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Graph saved at: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset folder containing JSON files')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the plot')
    args = parser.parse_args()

    # データセットを作成
    val_dataset = JSONDataset(data_dir=args.data_dir)
    dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # プロットを実行
    plot_true_acceleration(dataloader_val, save_folder=args.save_folder)