import os

class GlobalConfig:
    """ Base architecture configurations """

    # General Configurations
    seq_len = 1  # input timesteps
    img_seq_len = 1
    lidar_seq_len = 1
    pred_len = 4  # future waypoints predicted
    scale = 1  # image pre-processing
    img_resolution = (412, 1492)  # image pre-processing in H, W
    use_velocity = 1  # Include velocity data as input

    # LiDAR Configurations
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    # Target and Augmentation
    use_target_point_image = 0  # Disabled in provided settings
    gru_concat_target_point = False
    augment = True
    inv_augment_prob = 0.1  # Probability that augmentation is applied: 1.0 - inv_augment_prob
    aug_max_rotation = 20  # degree

    # Debugging
    debug = False
    train_debug_save_freq = 50

    # Perception and Backbone Configurations
    backbone = 'transFuser'  # Provided backbone
    image_architecture = 'resnet34'  # Image branch architecture
    lidar_architecture = 'resnet18'  # LiDAR branch architecture
    perception_output_features = 512  # Features from perception branch
    bev_features_chanels = 64
    bev_upsample_factor = 2

    # Detailed Loss Configurations
    detailed_losses = ['loss_acceleration']  # Adjusted to focus on acceleration
    detailed_losses_weights = [1.0, 1.0]

    # Optimization Configurations
    lr = 0.0001  # Updated learning rate
    multitask = False  # Only using regression for acceleration
    ls_acceleration = 1.0  # Weight for acceleration loss

    # Conv Encoder
    img_vert_anchors = 5
    img_horz_anchors = 22
    lidar_vert_anchors = 8
    lidar_horz_anchors = 8

    # GPT Encoder Configurations
    n_embd = 512
    block_exp = 4
    n_layer = 18  # Updated Transformer layer count
    n_head = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    # Training Parameters
    epochs = 10  # Updated epoch count
    batch_size = 4  # Updated batch size
    val_every = 5  # Validation frequency
    sync_batch_norm = False

    # Constructor
    def __init__(self, root_dir='', setting='all', **kwargs):
        self.root_dir = root_dir or "${WORK_DIR}/our_project/dataset/train_dataset"
        self.train_data = []
        self.val_data = []

        if setting == 'all':  # All towns used for training, no validation
            self.train_towns = os.listdir(self.root_dir)
            self.val_towns = [self.train_towns[0]]
            self._prepare_data()
        elif setting == 'eval':  # Evaluation only
            pass
        else:
            raise ValueError(f"Error: Selected setting '{setting}' does not exist.")

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _prepare_data(self, exclude_towns=None):
        exclude_towns = exclude_towns or []
        for town in os.listdir(self.root_dir):
            if any(exclude in town for exclude in exclude_towns):
                continue
            town_path = os.path.join(self.root_dir, town)
            if os.path.isdir(town_path):
                for file in os.listdir(town_path):
                    if file.endswith('.npy') or file.endswith('.png'):
                        self.train_data.append(os.path.join(town_path, file))
