import matplotlib.pyplot as plt
import os
import json

# データセットクラスを定義
class JSONDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(
            [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.json')]
        )  # JSONファイルを名前順にソート
        if not self.files:
            raise FileNotFoundError(f"No JSON files found in directory: {data_dir}")

    def load_data(self):
        basic_accelerations = []
        velocities = []
        for file in self.files:
            with open(file, 'r') as f:
                data = json.load(f)
            basic_accelerations.append(data.get('basic_acceleration', 0.0))
            velocities.append(data.get('velocity', 0.0))
        return basic_accelerations, velocities

# プロット関数を定義
def plot_basic_acceleration_and_velocity(basic_accelerations, velocities, save_folder):
    """
    basic_acceleration と velocity をプロットする関数
    既定の表示範囲:
        velocity:            0 〜 80
        basic_acceleration: -2.0 〜 2.0
    この範囲を超える値が存在する場合のみ，超えた分だけ軸を拡張する。
    """
    # ───────────────────────────────
    # 1) basic_acceleration のプロット
    # ───────────────────────────────
    acc_default_min, acc_default_max = -2.0, 2.0
    acc_data_min,     acc_data_max   = min(basic_accelerations), max(basic_accelerations)
    acc_ylim_min = min(acc_default_min, acc_data_min)
    acc_ylim_max = max(acc_default_max, acc_data_max)

    plt.figure(figsize=(10, 6))
    plt.plot(basic_accelerations, label='basic_acceleration', color='blue')
    plt.title('basic_acceleration Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('basic_acceleration (m/s²)')
    plt.ylim(acc_ylim_min, acc_ylim_max)  # ★ ここで動的に設定
    plt.legend()
    basic_acceleration_path = os.path.join(save_folder, 'basic_acceleration_plot.png')
    plt.savefig(basic_acceleration_path, dpi=150)
    print(f"basic_acceleration graph saved at: {basic_acceleration_path}")
    plt.close()

    # ───────────────────────────────
    # 2) velocity のプロット
    # ───────────────────────────────
    vel_default_min, vel_default_max = 0.0, 80.0
    vel_data_min,     vel_data_max   = min(velocities), max(velocities)
    vel_ylim_min = min(vel_default_min, vel_data_min)
    vel_ylim_max = max(vel_default_max, vel_data_max)

    plt.figure(figsize=(10, 6))
    plt.plot(velocities, label='Velocity', color='green')
    plt.title('Velocity Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Velocity (km/h)')
    plt.ylim(vel_ylim_min, vel_ylim_max)  # ★ ここで動的に設定
    plt.legend()
    velocity_path = os.path.join(save_folder, 'velocity_plot.png')
    plt.savefig(velocity_path, dpi=150)
    print(f"Velocity graph saved at: {velocity_path}")
    plt.close()


# 複数ディレクトリを処理する関数
def process_directories(base_dir):
    """
    指定した親ディレクトリ内の全てのサブディレクトリに対して処理を実行

    Parameters:
    - base_dir: str
        親ディレクトリのパス
    """
    subdirs = [os.path.join(base_dir, subdir) for subdir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subdir))]
    
    for subdir in subdirs:
        print(f"Processing directory: {subdir}")
        measurements_dir = os.path.join(subdir, 'measurements')
        if not os.path.exists(measurements_dir):
            print(f"Measurements directory not found in {subdir}. Skipping.")
            continue
        
        try:
            # データを読み込む
            dataset = JSONDataset(data_dir=measurements_dir)
            basic_accelerations, velocities = dataset.load_data()

            # プロットを実行
            plot_basic_acceleration_and_velocity(basic_accelerations, velocities, save_folder=subdir)
        except Exception as e:
            print(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process multiple directories for JSON plotting.")
    parser.add_argument('--base_dir', type=str, required=True, help='Path to the parent directory containing subdirectories')
    args = parser.parse_args()

    # 指定された親ディレクトリを処理
    process_directories(base_dir=args.base_dir)
