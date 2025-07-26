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
        accelerations = []
        velocities = []
        for file in self.files:
            with open(file, 'r') as f:
                data = json.load(f)
            accelerations.append(data.get('acceleration', 0.0))
            velocities.append(data.get('velocity', 0.0))
        return accelerations, velocities

# プロット関数を定義
def plot_acceleration_and_velocity(accelerations, velocities, save_folder):
    """
    accelerationとvelocityをプロットする関数

    Parameters:
    - accelerations: list
        accelerationのデータリスト
    - velocities: list
        velocityのデータリスト
    - save_folder: str
        グラフを保存するフォルダ
    """
    # accelerationのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(accelerations, label='Acceleration', color='blue')
    plt.title('Acceleration Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration')
    plt.legend()
    acceleration_path = os.path.join(save_folder, 'acceleration_plot.png')
    plt.savefig(acceleration_path, dpi=150)
    print(f"Acceleration graph saved at: {acceleration_path}")
    plt.close()

    # velocityのプロット
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, label='Velocity', color='green')
    plt.title('Velocity Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Velocity')
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
            accelerations, velocities = dataset.load_data()

            # プロットを実行
            plot_acceleration_and_velocity(accelerations, velocities, save_folder=subdir)
        except Exception as e:
            print(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process multiple directories for JSON plotting.")
    parser.add_argument('--base_dir', type=str, required=True, help='Path to the parent directory containing subdirectories')
    args = parser.parse_args()

    # 指定された親ディレクトリを処理
    process_directories(base_dir=args.base_dir)
