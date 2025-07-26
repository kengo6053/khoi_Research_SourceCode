import os
from copy import deepcopy
import json
import torch
from torch.utils.data import Dataset, DataLoader
from model_stereo import LidarCenterNet  # あなたのモデルクラスに合わせてください
from config_stereo import GlobalConfig    # あなたの設定クラスに合わせてください
from PIL import Image
from data import lidar_to_histogram_features
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime


class StereoData(Dataset):
    def __init__(self, root, config, save_pc_bev=False, save_dir='./'):
        """
        Args:
            root (str): 走行フォルダ (例: EX04AM03 など)
            config (GlobalConfig): グローバル設定
            save_pc_bev (bool): prepared_point_cloud_bev を保存するかどうか
            save_dir (str): 保存先ディレクトリ
        """
        self.root = root
        self.config = config
        self.img_height, self.img_width = config.img_resolution
        self.save_pc_bev = save_pc_bev
        self.save_dir = save_dir
        self.saved = False  # 一度だけ保存するためのフラグ

        # データを格納するリスト
        self.images = []
        self.point_clouds = []
        self.velocities = []
        self.true_basic_accelerations = []

        # 走行フォルダ直下にあるはずのサブフォルダ
        gray_dir = os.path.join(self.root, "gray_right")
        point_cloud_dir = os.path.join(self.root, "point_cloud_transformed")
        measurements_dir = os.path.join(self.root, "measurements")

        # 必要なサブフォルダが揃っているかチェック
        if not (os.path.isdir(gray_dir) and
                os.path.isdir(point_cloud_dir) and
                os.path.isdir(measurements_dir)):
            print(f"Missing subfolders in '{self.root}'. "
                  "Expected 'gray', 'point_cloud_transformed', 'measurements'.")
            return  # 空のリストのまま返す

        # gray フォルダ内の RAW 画像をソートして取得
        raw_files = sorted([f for f in os.listdir(gray_dir) if f.endswith('.raw')])
        for f in raw_files:
            # 拡張子を除いた部分 (例: "0000") を取り出す
            idx_str = os.path.splitext(f)[0]

            img_path = os.path.join(gray_dir, f"{idx_str}.raw")
            pc_path = os.path.join(point_cloud_dir, f"{idx_str}.npy")
            meas_path = os.path.join(measurements_dir, f"{idx_str}.json")

            # 必要ファイルの存在チェック
            if os.path.isfile(img_path) and os.path.isfile(pc_path) and os.path.isfile(meas_path):
                self.images.append(img_path)
                self.point_clouds.append(pc_path)
                with open(meas_path, 'r') as mf:
                    measurement = json.load(mf)
                    self.velocities.append(measurement.get("velocity", 0.0))
                    self.true_basic_accelerations.append(measurement.get("basic_acceleration", 0.0))
            else:
                print(f"Missing file(s) in '{self.root}': {idx_str}.raw / .npy / .json")

        # point_cloud_bev を保存するディレクトリを作成 (オプション)
        if self.save_pc_bev:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        gray_image = self.read_raw_image(self.images[idx],
                                         self.img_height,
                                         self.img_width)
        prepared_image = self.prepare_image(gray_image)

        point_cloud = np.load(self.point_clouds[idx], allow_pickle=True)
        # 特定の構造 (2要素の配列) で格納されている場合は修正
        if point_cloud.shape == (2,) and isinstance(point_cloud[1], np.ndarray):
            point_cloud = point_cloud[1]
        if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
            raise ValueError(f"Invalid LiDAR data shape at index {idx} in '{self.root}': "
                             f"{point_cloud.shape}. Expected (N, 3) or (N, 4).")

        prepared_point_cloud_bev = self.prepare_point_cloud(point_cloud)

        # 必要なら一度だけ BEV データを保存
        if self.save_pc_bev and not self.saved:
            save_path = os.path.join(self.save_dir, f"prepared_point_cloud_bev_{idx}.npy")
            np.save(save_path, prepared_point_cloud_bev.cpu().numpy())
            print(f"Saved prepared_point_cloud_bev to {save_path}")
            self.saved = True

        velocity = torch.tensor(self.velocities[idx], dtype=torch.float32).unsqueeze(0)
        true_basic_acceleration = torch.tensor(self.true_basic_accelerations[idx], dtype=torch.float32).unsqueeze(0)

        return {
            "image": prepared_image,
            "point_cloud": prepared_point_cloud_bev,
            "velocity": velocity,
            "true_basic_acceleration": true_basic_acceleration
        }

    def read_raw_image(self, filepath, height, width):
        """RAWファイルを読み込んで (height, width) の2次元配列として返す"""
        try:
            with open(filepath, 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint8)
            image = raw_data.reshape((height, width))
            return image
        except Exception as e:
            print(f"Error reading RAW file {filepath}: {e}")
            return np.zeros((height, width), dtype=np.uint8)  # 読み込めない場合は0配列

    def prepare_image(self, gray_image):
        """画像前処理 (リサイズ・正規化・テンソル化)"""
        image = Image.fromarray(gray_image)
        image = image.resize((self.img_width, self.img_height))
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return image

    def prepare_point_cloud(self, point_cloud):
        """点群をBEV変換し、テンソルにする"""
        features = lidar_to_histogram_features(deepcopy(point_cloud))
        point_cloud_tensor = torch.from_numpy(features).float()
        if point_cloud_tensor.shape[0] != 2:
            raise ValueError(f"Expected 2 channels in point_cloud_tensor, but got {point_cloud_tensor.shape[0]}")
        return point_cloud_tensor


def main(config_dir, dataset_path, output_folder, allowed_folders_file=None):
    """
    走行フォルダごとに StereoData を作成し、推定を行い、MSE を計算する。
    txtファイルに記載されたフォルダ名のみを対象とするように修正。
    """
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parent_output_folder = os.path.join(output_folder, f"inference_results_{timestamp}")
    os.makedirs(parent_output_folder, exist_ok=True)

    # グローバル設定ファイル (args.txt) の読み込み
    args_path = os.path.join(config_dir, "args.txt")
    with open(args_path, "r") as args_file:
        args = json.load(args_file)

    # グローバル設定に画像サイズなどを反映
    config = GlobalConfig(setting="eval")
    config.img_resolution = (
        args.get("img_height", 412),
        args.get("img_width", 1492)
    )

    # allowed_folders_file が指定されていれば、その内容を読み込む
    allowed_folders = None
    if allowed_folders_file is not None:
        with open(allowed_folders_file, "r") as f:
            allowed_folders = [line.strip() for line in f if line.strip()]

    if allowed_folders is not None:
        print("Allowed folders to process:", allowed_folders)


    all_mse_values = []
    subdirs_processed = []

    # dataset_path 下にある各走行フォルダを探索
    for subdir in os.listdir(dataset_path):
        # allowed_folders が指定されている場合、そのリストに含まれるフォルダのみを処理する
        if allowed_folders is not None and subdir not in allowed_folders:
            continue

        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            continue  # フォルダでないならスキップ

        print(f"Processing directory: {subdir}")

        # 推論結果を保存するフォルダ
        sub_output_folder = os.path.join(parent_output_folder, subdir)
        os.makedirs(sub_output_folder, exist_ok=True)

        # StereoData インスタンス作成
        dataset = StereoData(
            root=subdir_path,
            config=config,
            save_pc_bev=True,
            save_dir=sub_output_folder
        )

        # データが0件ならスキップ
        if len(dataset) == 0:
            print(f"No valid data found in directory '{subdir}'. Skipping MSE calculation.")
            continue

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        # モデルを全て読み込む (複数あればそれぞれ推論し、平均化)
        model_list = []
        for file in os.listdir(config_dir):
            if file.endswith(".pth"):
                model_path = os.path.join(config_dir, file)
                print(f"Loading model from: {model_path}")

                model = LidarCenterNet(
                    config=config,
                    device=device,
                    backbone=args.get("backbone", "transFuser"),
                    image_architecture=args.get("image_architecture", "resnet34"),
                    point_cloud_architecture=args.get("point_cloud_architecture", "resnet18"),
                    use_velocity=args.get("use_velocity", 1)
                ).to(device)

                state_dict = torch.load(model_path, map_location=device)
                # "module." がついているキーを除去
                state_dict = {
                    k.replace("module.", ""): v
                    for k, v in state_dict.items()
                }
                model.load_state_dict(state_dict)
                model.eval()
                model_list.append(model)

        # 推論・MSE計算
        predictions_list = []
        true_basic_accelerations_list = []

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                image = batch["image"].to(device)
                point_cloud = batch["point_cloud"].to(device)
                velocity = batch["velocity"].to(device)
                # バッチサイズは1なので .item() で値を取得
                true_basic_acceleration = batch["true_basic_acceleration"].item()

                # 複数モデルがある場合は予測を平均化
                predictions = [m(image, point_cloud, velocity).item() for m in model_list]
                avg_prediction = np.mean(predictions)

                predictions_list.append(avg_prediction)
                true_basic_accelerations_list.append(true_basic_acceleration)

        # 推論結果が空ならスキップ
        if len(predictions_list) == 0:
            print(f"No predictions were made for directory '{subdir}'. Skipping MSE calculation.")
            continue

        # MSE 計算
        mse = mean_squared_error(true_basic_accelerations_list, predictions_list)
        all_mse_values.append(mse)
        subdirs_processed.append(subdir)
        print(f"Mean Squared Error (MSE) for {subdir}: {mse}")

        # 結果を JSON として保存
        results_file = os.path.join(sub_output_folder, "predicted_results.json")
        with open(results_file, "w") as out_file:
            json.dump({
                "predictions": predictions_list,
                "true_values": true_basic_accelerations_list,
                "MSE": mse
            }, out_file, indent=2)
        print(f"Results saved to {results_file}")

        # 可視化してプロット保存
        plt.figure(figsize=(10, 6))
        plt.plot(true_basic_accelerations_list, label="True Acceleration", color="blue")
        plt.plot(predictions_list, label="Predicted Acceleration", color="red", linestyle="--")
        plt.xlabel("Frame")
        plt.ylabel("Acceleration")
        plt.title(f"Predicted vs True Acceleration for {subdir}")
        plt.legend()
        
        # デフォルトの表示範囲
        y_min, y_max = -2.0, 2.0

        # 真の加速度と予測値の最小値・最大値を取得
        all_values = np.concatenate([true_basic_accelerations_list, predictions_list])
        data_min, data_max = all_values.min(), all_values.max()

        # 範囲外があれば拡張
        if data_min < y_min:
            y_min = data_min - 0.5  # 余裕を持たせるために -0.5
        if data_max > y_max:
            y_max = data_max + 0.5  # 余裕を持たせるために +0.5

        # y 軸範囲を設定
        plt.ylim(y_min, y_max)
        
        plot_file = os.path.join(sub_output_folder, "acceleration_plot.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")

    # 全ディレクトリの平均 MSE をまとめて表示・保存
    if len(all_mse_values) > 0:
        average_mse = np.mean(all_mse_values)
    else:
        average_mse = float("nan")

    summary_file = os.path.join(parent_output_folder, "mse_summary.txt")
    with open(summary_file, "w") as summary:
        summary.write("MSE Results for Each Directory:\n")
        for subdir, mse in zip(subdirs_processed, all_mse_values):
            summary.write(f"{subdir}: {mse:.4f}\n")
        summary.write(f"\nAverage MSE: {average_mse:.4f}\n")

    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, help="Path to configuration directory")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--output_folder", required=True, help="Output folder for results and plots")
    parser.add_argument("--allowed_folders_file", required=False, help="Path to txt file containing allowed folder names")
    args = parser.parse_args()

    main(args.config_dir, args.dataset_path, args.output_folder, allowed_folders_file=args.allowed_folders_file)
