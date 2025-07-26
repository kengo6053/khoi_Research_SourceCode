import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold

import pathlib
import datetime
import random
import cv2
from PIL import Image
from copy import deepcopy
import timm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================
# 追加: DDP 関連の import
# ========================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config_stereo import GlobalConfig
from model_stereo_par import LidarCenterNet  # DDP 対応の model_stereo_par を想定
from data import lidar_to_histogram_features

# ----------------------------------------------------------------------------
# 追加: RegNet / ResNet は torchvision、ConvNeXt 等は timm を使うかどうか判定する関数
# ----------------------------------------------------------------------------
def is_torchvision_arch(arch_name: str) -> bool:
    """
    簡易的に、'regnet_y_' または 'resnet' で始まるモデル名は torchvision で扱うとする。
    必要に応じて条件を追加してください。
    """
    arch_name_lower = arch_name.lower()
    if arch_name_lower.startswith("regnet_y_"):
        return True
    if arch_name_lower.startswith("resnet"):
        return True
    return False

class StereoData(Dataset):
    def __init__(self, root, config, device, split='train', shared_dict=None, save_pc_bev=False, save_dir='./', allowed_folders=None):
        """
        Args:
            root (str): データセットのルートディレクトリ
            config (GlobalConfig): グローバル設定
            device (torch.device): デバイス
            split (str): 'train', 'val', 'test' のいずれか
            shared_dict (dict): データの共有辞書（オプション）
            save_pc_bev (bool): prepared_point_cloud_bev を保存するか
            save_dir (str): prepared_point_cloud_bev を保存するディレクトリ
            allowed_folders (list): 読み込むフォルダ名のリスト (例: ['EX00AM00', 'EX00AM01', ...])
        """
        self.root = root
        self.config = config
        self.device = device
        self.split = split
        self.shared_dict = shared_dict
        self.img_height, self.img_width = config.img_resolution
        self.images = []
        self.point_clouds = []
        self.velocities = []
        self.accelerations = []
        self.save_pc_bev = save_pc_bev  # 保存フラグ
        self.save_dir = save_dir  # 保存ディレクトリ
        self.saved = False  # 一度だけ保存するためのフラグ
        self.allowed_folders = allowed_folders  # 許可されたフォルダ名のリスト

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for route in os.listdir(split_dir):
            # フォルダフィルタリング (allowed_folders が None の場合はスキップ)
            if self.allowed_folders and route not in self.allowed_folders:
                continue

            route_path = os.path.join(split_dir, route)
            if os.path.isdir(route_path):
                gray_dir = os.path.join(route_path, "gray_right")
                point_cloud_dir = os.path.join(route_path, "point_cloud_transformed")
                measurements_dir = os.path.join(route_path, "measurements")

                if not (os.path.isdir(gray_dir) and os.path.isdir(point_cloud_dir) and os.path.isdir(measurements_dir)):
                    continue

                num_files = len(os.listdir(gray_dir))
                for idx in range(num_files):
                    img_path = os.path.join(gray_dir, f"{idx:04d}.raw")
                    pc_path = os.path.join(point_cloud_dir, f"{idx:04d}.npy")
                    meas_path = os.path.join(measurements_dir, f"{idx:04d}.json")

                    if os.path.isfile(img_path) and os.path.isfile(pc_path) and os.path.isfile(meas_path):
                        self.images.append(img_path)
                        self.point_clouds.append(pc_path)
                        with open(meas_path, 'r') as f:
                            measurement = json.load(f)
                            self.velocities.append(measurement.get("velocity", 0.0))
                            self.accelerations.append(measurement.get("basic_acceleration", 0.0))

        # 保存ディレクトリの作成
        if self.save_pc_bev:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        gray_image = self.read_raw_image(self.images[idx], self.img_height, self.img_width)
        if gray_image is None:
            raise FileNotFoundError(f"Image file not found: {self.images[idx]}")
        prepared_image = self.prepare_image(gray_image)

        point_cloud = np.load(self.point_clouds[idx], allow_pickle=True)
        if point_cloud.shape == (2,) and isinstance(point_cloud[1], np.ndarray):
            point_cloud = point_cloud[1]

        if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
            raise ValueError(f"Invalid LiDAR data shape at index {idx}: {point_cloud.shape}. "
                             "Expected a (N, 3) or (N, 4) array.")
        
        # ==== 座標の入れ替え処理 ====
        # 点群データの x と y を入れ替えます
        #point_cloud[:, [0, 1]] = point_cloud[:, [1, 0]]
        # =========================
        #point_cloud[:, 1] = point_cloud[:, 1] * -1

        prepared_point_cloud_bev = self.prepare_point_cloud(point_cloud)

        # 最初のデータポイントでのみ保存
        if self.save_pc_bev and not self.saved:
            save_path = os.path.join(self.save_dir, "prepared_point_cloud_bev.npy")
            np.save(save_path, prepared_point_cloud_bev.cpu().numpy())
            print(f"Saved prepared_point_cloud_bev to {save_path}")
            self.saved = True  # 一度だけ保存するためにフラグを立てる

        velocity = torch.tensor(self.velocities[idx], dtype=torch.float32)
        acceleration = torch.tensor(self.accelerations[idx], dtype=torch.float32)

        return {
            "image": prepared_image,
            "point_cloud": prepared_point_cloud_bev,
            "velocity": velocity.unsqueeze(0),
            "acceleration": acceleration
        }

    def read_raw_image(self, filepath, height, width):
        try:
            with open(filepath, 'rb') as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint8)
            image = raw_data.reshape((height, width))
            return image
        except Exception as e:
            print(f"Error reading RAW file {filepath}: {e}")
            return None

    def prepare_image(self, gray_image):
        image = Image.fromarray(gray_image)
        image = image.resize((self.img_width, self.img_height))
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return image

    def prepare_point_cloud(self, point_cloud):
        features = lidar_to_histogram_features(deepcopy(point_cloud))
        point_cloud_tensor = torch.from_numpy(features).float()
        assert point_cloud_tensor.shape[0] == 2, f"Expected 2 channels, got {point_cloud_tensor.shape[0]} channels"
        return point_cloud_tensor

class Engine(object):
    def __init__(self, model, optimizer, dataloader_train, dataloader_val, args, config, writer, device, cur_epoch=0, checkpoint_dir=None):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.args = args
        self.config = config
        self.writer = writer
        self.device = device

        # チェックポイントやグラフを保存するフォルダ
        self.checkpoint_dir = checkpoint_dir

        if not hasattr(self.config, 'debug'):
            self.config.debug = False
        if not hasattr(self.config, 'detailed_losses'):
            self.config.detailed_losses = ['loss_total']
        if not hasattr(self.config, 'detailed_losses_weights'):
            self.config.detailed_losses_weights = [1.0]

        self.vis_save_path = os.path.join(self.args.logdir, 'visualizations')
        if self.config.debug:
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_losses = self.config.detailed_losses
        if 'loss_total' not in self.detailed_losses:
            self.detailed_losses.append('loss_total')
            self.config.detailed_losses_weights.append(1.0)

        if self.args.wp_only:
            detailed_losses_weights = [1.0] + [0.0]*(len(self.detailed_losses)-1)
        else:
            detailed_losses_weights = self.config.detailed_losses_weights
        self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

    def load_data_compute_loss(self, data):
        image = data['image'].to(self.device, dtype=torch.float32)
        point_cloud = data['point_cloud'].to(self.device, dtype=torch.float32)
        velocity = data['velocity'].to(self.device, dtype=torch.float32)
        acceleration = data['acceleration'].to(self.device, dtype=torch.float32)

        outputs = self.model(image, point_cloud, velocity)

        # デバッグ用
        print("Model Outputs:", outputs)
        print("Targets (Acceleration):", acceleration)

        loss_dict = self.compute_loss(outputs, acceleration)
        return loss_dict

    def compute_loss(self, outputs, targets):
        targets = targets.unsqueeze(-1)
        loss_acceleration = torch.nn.functional.mse_loss(outputs, targets, reduction='mean')
        loss_total = loss_acceleration
        return {
            "loss_total": loss_total,
            "loss_acceleration": loss_acceleration
        }

    def train(self):
        self.model.train()
        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_losses}
        self.cur_epoch += 1

        # DDP 使用時には epoch 開始時にサンプラーを再初期化
        if isinstance(self.dataloader_train.sampler, DistributedSampler):
            self.dataloader_train.sampler.set_epoch(self.cur_epoch)

        for data in tqdm(self.dataloader_train, desc=f"Training Epoch {self.cur_epoch}"):
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.load_data_compute_loss(data)
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for key, value in losses.items():
                w = self.detailed_weights.get(key, 1.0)
                total_loss += w * value
                detailed_losses_epoch[key] += float((w * value).item())
            total_loss.backward()
            self.optimizer.step()
            num_batches += 1
            loss_epoch += float(total_loss.item())

        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, prefix='train_')

    @torch.inference_mode()
    def validate(self):
        if len(self.dataloader_val.dataset) == 0:
            print(f"[Warning] Validation dataset is empty at epoch {self.cur_epoch}. Skipping validation.")
            return

        self.model.eval()
        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch = {key: 0.0 for key in self.detailed_losses}

        all_preds = []
        all_trues = []

        for data in tqdm(self.dataloader_val, desc=f"Validation Epoch {self.cur_epoch}"):
            losses = self.load_data_compute_loss(data)
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for key, value in losses.items():
                w = self.detailed_weights.get(key, 1.0)
                total_loss += w * value
                detailed_val_losses_epoch[key] += float((w * value).item())
            num_batches += 1
            loss_epoch += float(total_loss.item())

            # 推論結果を保存
            image = data['image'].to(self.device, dtype=torch.float32)
            point_cloud = data['point_cloud'].to(self.device, dtype=torch.float32)
            velocity = data['velocity'].to(self.device, dtype=torch.float32)
            acceleration = data['acceleration'].to(self.device, dtype=torch.float32)

            outputs = self.model(image, point_cloud, velocity)
            preds = outputs.squeeze(-1).detach().cpu().numpy()
            trues = acceleration.detach().cpu().numpy()

            all_preds.append(preds)
            all_trues.append(trues)

        # ここから rank=0 のみ可視化やログを出力
        if dist.get_rank() == 0:
            print(f"Validation Epoch {self.cur_epoch} - Loss: {loss_epoch}")
            print(f"Validation Detailed Losses: {detailed_val_losses_epoch}")
            self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, prefix='val_')

            # np.concatenate も rank=0 だけでやる
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)

            # 必要に応じてディレクトリを先に作る
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # 可視化
            plt.figure(figsize=(8, 5))
            plt.plot(all_trues, color='blue', label='True Acceleration')
            plt.plot(all_preds, color='red', linestyle='--', label='Predicted Acceleration')
            plt.title('Predicted vs True Acceleration')
            plt.xlabel('Frame')
            plt.ylabel('Acceleration')
            plt.legend()

            plot_path = os.path.join(self.checkpoint_dir, f"pred_vs_true_epoch_{self.cur_epoch}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
        else:
            # rank!=0 はファイル保存せずに終了
            pass

    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        if num_batches == 0:
            print(f"[Warning] No batches found for prefix='{prefix}'. Skipping loss logging.")
            return

        loss_epoch /= num_batches
        for key in detailed_losses_epoch:
            detailed_val = detailed_losses_epoch[key] / num_batches
            detailed_losses_epoch[key] = detailed_val

        if self.writer:
            self.writer.add_scalar(prefix + 'loss_total', loss_epoch, self.cur_epoch)
            for key, value in detailed_losses_epoch.items():
                self.writer.add_scalar(prefix + key, value, self.cur_epoch)

        print(f"Aggregated Loss at Epoch {self.cur_epoch}: {prefix}loss_total = {loss_epoch}")
        for key, value in detailed_losses_epoch.items():
            print(f"  {prefix}{key} = {value}")

        if prefix.startswith('train_'):
            self.train_loss.append(loss_epoch)
        elif prefix.startswith('val_'):
            self.val_loss.append(loss_epoch)

    def save(self):
        model_path = os.path.join(self.checkpoint_dir, f'model_{self.cur_epoch}.pth')
        optim_path = os.path.join(self.checkpoint_dir, f'optimizer_{self.cur_epoch}.pth')

        # DDP を使用している場合は model.module.state_dict() を保存
        state_dict = (
            self.model.module.state_dict()
            if hasattr(self.model, 'module')
            else self.model.state_dict()
        )
        torch.save(state_dict, model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

def main():
    parser = argparse.ArgumentParser(
        description="Train LidarCenterNet on CARLA Dataset (DDP + K-fold + Random Search)"
    )

    # 環境変数 LOCAL_RANK があればそれを使う。なければ -1 (単独実行時対策)
    local_rank_env = int(os.environ.get("LOCAL_RANK", -1))
    parser.add_argument("--local_rank", type=int, default=local_rank_env)

    # 基本パラメータ
    parser.add_argument('--id', type=str, default='transfuser')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--logdir', type=str, default='./output_log')
    parser.add_argument('--load_file', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--setting', type=str, default='all')
    parser.add_argument('--root_dir', type=str, default='./dataset/train_dataset')
    parser.add_argument('--schedule', type=int, default=1)
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=10)
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=20)
    parser.add_argument('--backbone', type=str, default='transFuser')
    parser.add_argument('--image_architecture', type=str, default='resnet34')
    parser.add_argument('--lidar_architecture', type=str, default='resnet18')
    parser.add_argument('--use_velocity', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=18)
    parser.add_argument('--wp_only', type=int, default=0)
    parser.add_argument('--use_target_point_image', type=int, default=0)
    parser.add_argument('--use_point_pillars', type=int, default=0)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--no_bev_loss', type=int, default=0)
    parser.add_argument('--use_disk_cache', type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--embd_pdrop', type=float, default=0.1, help='Embedding dropout probability')
    parser.add_argument('--resid_pdrop', type=float, default=0.1, help='Residual dropout probability')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='Attention dropout probability')



    # ===== ランダムサーチ関連オプション =====
    parser.add_argument('--random_search', type=int, default=0,
                        help='ランダムサーチを行うかどうか (0: しない, 1: する)')
    parser.add_argument('--random_search_iters', type=int, default=3,
                        help='ランダムサーチの試行回数')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='データセットの一部だけを使う場合(0~1)。1.0は全データを使用')

    parser.add_argument('--allowed_folders_file', type=str, default=None, help='Path to the file containing allowed folders')

    args = parser.parse_args()

    # ===== 分散処理の初期化 =====
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[INFO] Process initialized. Local Rank: {args.local_rank}, Global Rank: {rank}")

    # 許可されたフォルダのリストを読み込む
    if args.allowed_folders_file:
        with open(args.allowed_folders_file, 'r') as f:
            allowed_folders = [line.strip() for line in f.readlines()]
    else:
        allowed_folders = None  # 制限なしの場合
        
    # ランダムサーチをしない場合 (random_search=0) → 1回だけ実験して終了
    if args.random_search == 0:
        val_loss_mean = run_experiment_once(args, device, allowed_folders)
        dist.destroy_process_group()  # 全プロセス終了
        return
    
    # ======================================================
    # ランダムサーチを行う場合 (random_search=1)
    # ======================================================
    lr_candidates = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    batch_size_candidates = [4, 8, 10]
    epochs_candidates = [20]

    # torchvision.models で利用可能なアーキテクチャ名 (例)
    #arch_candidates = [
    #    'regnety_032',
    #    'regnety_016',
    #    'regnety_008',
    #    #'convnext_tiny',
    #    'resnet34',
    #   'resnet18'
    #]
    
    embd_pdrop_candidates = [0.1, 0.2, 0.3]
    resid_pdrop_candidates = [0.1, 0.2, 0.3]
    attn_pdrop_candidates = [0.1, 0.2, 0.3]
    inv_augment_prob_candidates = [0.1, 0.3, 0.5]
    aug_max_rotation_candidates = [10, 20, 30]
    detailed_losses_weights_candidates = [1.0, 5.0, 10.0]

    best_val_loss = float('inf')
    best_config = None

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    random_search_root = os.path.join(args.logdir, args.id, f"random_search_{timestamp}")

    if rank == 0:
        os.makedirs(random_search_root, exist_ok=True)

    dist.barrier()  # 全 rank 同期

    for i in range(args.random_search_iters):
        if rank == 0:
            chosen_config = {
                "lr": random.choice(lr_candidates),
                "batch_size": random.choice(batch_size_candidates),
                "epochs": random.choice(epochs_candidates),
                #"image_arch": random.choice(arch_candidates),
                #"lidar_arch": None, 
                "embd_pdrop": random.choice(embd_pdrop_candidates),
                "resid_pdrop": random.choice(resid_pdrop_candidates),
                "attn_pdrop": random.choice(attn_pdrop_candidates),
                "inv_augment_prob": random.choice(inv_augment_prob_candidates),
                "aug_max_rotation": random.choice(aug_max_rotation_candidates),
                "detailed_losses_weights": random.choice(detailed_losses_weights_candidates)
            }
            # image_arch と同じものを lidar_arch にする
            #chosen_config["lidar_arch"] = chosen_config["image_arch"]
        else:
            chosen_config = {}
            
        dist.barrier()
        obj_list = [chosen_config]
        dist.broadcast_object_list(obj_list, src=0)
        chosen_config = obj_list[0]

        lr_val = chosen_config["lr"]
        batch_sz_val = chosen_config["batch_size"]
        epochs_val = chosen_config["epochs"]
        #image_arch_val = chosen_config["image_arch"]
        #lidar_arch_val = chosen_config["lidar_arch"]
        embd_pdrop_val = chosen_config["embd_pdrop"]
        resid_pdrop_val = chosen_config["resid_pdrop"]
        attn_pdrop_val = chosen_config["attn_pdrop"]
        inv_augment_prob_val = chosen_config["inv_augment_prob"]
        aug_max_rotation_val = chosen_config["aug_max_rotation"]
        dloss_weight_val = chosen_config["detailed_losses_weights"]

        if rank == 0:
            print(f"\n===== Random Search Trial {i+1}/{args.random_search_iters} =====")
            print(f"  - lr={lr_val}, batch_size={batch_sz_val}, epochs={epochs_val}")
            #print(f"  - image_arch={image_arch_val}, lidar_arch={lidar_arch_val}")
            print(f"  - embd_pdrop={embd_pdrop_val}, resid_pdrop={resid_pdrop_val}, attn_pdrop={attn_pdrop_val}")
            print(f"  - inv_augment_prob={inv_augment_prob_val}, aug_max_rotation={aug_max_rotation_val}")
            print(f"  - detailed_losses_weights={dloss_weight_val}")

        trial_args = argparse.Namespace(**vars(args))
        trial_args.lr = lr_val
        trial_args.batch_size = batch_sz_val
        trial_args.epochs = epochs_val
        #trial_args.image_architecture = image_arch_val
        #trial_args.lidar_architecture = lidar_arch_val

        trial_args.embd_pdrop = embd_pdrop_val
        trial_args.resid_pdrop = resid_pdrop_val
        trial_args.attn_pdrop = attn_pdrop_val
        trial_args.inv_augment_prob = inv_augment_prob_val
        trial_args.aug_max_rotation = aug_max_rotation_val
        trial_args.detailed_losses_weights = dloss_weight_val

        trial_logdir = os.path.join(random_search_root, f"random_search_trial_{i+1}")
        if rank == 0:
            os.makedirs(trial_logdir, exist_ok=True)

        dist.barrier()
        val_loss_mean = run_experiment_once(trial_args, device, custom_logdir=trial_logdir)

        if rank == 0:
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                best_config = {
                    'lr': lr_val,
                    'batch_size': batch_sz_val,
                    'epochs': epochs_val,
                    #'image_architecture': image_arch_val,
                    #'lidar_architecture': lidar_arch_val,
                    'embd_pdrop': embd_pdrop_val,
                    'resid_pdrop': resid_pdrop_val,
                    'attn_pdrop': attn_pdrop_val,
                    'inv_augment_prob': inv_augment_prob_val,
                    'aug_max_rotation': aug_max_rotation_val,
                    'detailed_losses_weights': dloss_weight_val,
                }

    if rank == 0:
        print("\n======== ランダムサーチ結果まとめ ========")
        print(f"最良のバリデーション損失: {best_val_loss:.4f}")
        print("最良設定:", best_config)

    dist.destroy_process_group()

def run_experiment_once(args, device, allowed_folders=None, custom_logdir=None):
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
    config.use_velocity = bool(args.use_velocity)
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone

    ## 追加: torchvision / timm をどちらを使うか設定する
    #config.use_torchvision_for_image = is_torchvision_arch(args.image_architecture)
    #config.use_torchvision_for_lidar = is_torchvision_arch(args.lidar_architecture)

    ## デバッグ用ログ
    #if dist.get_rank() == 0:
    #    print(f"[DEBUG] image_architecture={args.image_architecture}, use_torchvision_for_image={config.use_torchvision_for_image}")
    #    print(f"[DEBUG] lidar_architecture={args.lidar_architecture}, use_torchvision_for_lidar={config.use_torchvision_for_lidar}")

    # データセット読み込み時に allowed_folders を渡す
    full_dataset = StereoData(
        root=args.root_dir,
        config=config,
        device=device,
        split='train',
        allowed_folders=allowed_folders  # フィルタリング用フォルダリスト
    )
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    use_count = int(len(indices) * args.sample_rate)
    sampled_indices = indices[:use_count]
    dataset_sampled = Subset(full_dataset, sampled_indices)

    dist.barrier()

    if custom_logdir is not None:
        args.logdir = custom_logdir
    else:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.logdir = os.path.join(args.logdir, args.id, current_time)
        if dist.get_rank() == 0:
            os.makedirs(args.logdir, exist_ok=True)
    dist.barrier()

    if dist.get_rank() == 0:
        with open(os.path.join(args.logdir, f'args_rank_{dist.get_rank()}.txt'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_sampled)):
        if dist.get_rank() == 0:
            print(f"\n========== Fold {fold+1}/{args.num_folds} (Rank={dist.get_rank()}) ==========")
            print(f"  -> train size: {len(train_indices)}, val size: {len(val_indices)}")

        if len(val_indices) == 0:
            if dist.get_rank() == 0:
                print(f"[Warning] Fold {fold+1}: No validation data. Setting fold_loss=inf.")
                fold_results.append(float('inf'))
            continue

        train_subset = Subset(dataset_sampled, train_indices)
        val_subset = Subset(dataset_sampled, val_indices)

        train_sampler = DistributedSampler(train_subset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        val_sampler = DistributedSampler(val_subset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

        dataloader_train = DataLoader(train_subset, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=4, pin_memory=True)
        dataloader_val = DataLoader(val_subset, batch_size=args.batch_size, sampler=val_sampler,
                                    num_workers=4, pin_memory=True)

        # ---------------------------------------------
        # LidarCenterNet の生成 (バックボーンの作成方式を内部で切り替える想定)
        # ---------------------------------------------
        model = LidarCenterNet(
            config=config,
            device=device,
            backbone=args.backbone,
            image_architecture=args.image_architecture,
            point_cloud_architecture=args.lidar_architecture,
            use_velocity=config.use_velocity
        ).to(device)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        if args.load_file and dist.get_rank() == 0:
            state_dict = torch.load(args.load_file, map_location="cpu")
            model.module.load_state_dict(state_dict)
            opt_path = args.load_file.replace("model_", "optimizer_")
            if os.path.isfile(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        dist.barrier()

        fold_logdir = os.path.join(args.logdir, f"fold_{fold+1}")
        if dist.get_rank() == 0:
            os.makedirs(fold_logdir, exist_ok=True)

        writer = SummaryWriter(log_dir=os.path.join(fold_logdir, "tensorboard")) if dist.get_rank() == 0 else None
        checkpoint_dir = os.path.join(fold_logdir, "checkpoints")
        if dist.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        trainer = Engine(
            model=model,
            optimizer=optimizer,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            args=args,
            config=config,
            writer=writer,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        for epoch in range(trainer.cur_epoch, args.epochs):
            if epoch in [args.schedule_reduce_epoch_01, args.schedule_reduce_epoch_02] and args.schedule:
                new_lr = optimizer.param_groups[0]['lr'] * 0.1
                if dist.get_rank() == 0:
                    print(f"Reduce learning rate by factor 10 to: {new_lr}")
                for g in optimizer.param_groups:
                    g['lr'] = new_lr

            trainer.train()
            if (epoch + 1) % args.val_every == 0:
                trainer.validate()
                if dist.get_rank() == 0:
                    trainer.save()

        trainer.validate()
        if dist.get_rank() == 0:
            trainer.save()

        if dist.get_rank() == 0 and len(trainer.val_loss) > 0:
            fold_results.append(trainer.val_loss[-1])

    if dist.get_rank() == 0:
        if len(fold_results) == 0:
            mean_loss = float('inf')
        else:
            mean_loss = np.mean(fold_results)
        print("\n========== K-Fold 結果まとめ ==========")
        for i, result in enumerate(fold_results):
            print(f"Fold {i+1}: {result:.4f}")
        print(f"平均: {mean_loss:.4f}, 分散: {np.var(fold_results):.4f}")
        return mean_loss
    else:
        return 9999.99

if __name__ == "__main__":
    main()
