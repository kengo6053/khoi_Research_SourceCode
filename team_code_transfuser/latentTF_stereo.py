import math
import torch
from torch import nn
import torch.nn.functional as F
import timm

def normalize_grayscale(x: torch.Tensor) -> torch.Tensor:
    """
    グレースケール画像を [0,1] の範囲に正規化する。
    Args:
        x (tensor): shape = [B, 1, H, W] のグレースケール画像
    """
    x = x.clone()
    return x / 255.0  # [0,255] → [0,1]

class latentTFBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + positional encoding (LiDAR) feature fusion.
    This version is modified similarly to how 'transfuser_stereo.py' differs from 'transfuser.py',
    i.e. using grayscale image input, simpler LiDAR in_channels=2, etc.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=1):
        super().__init__()
        self.config = config

        # 画像/LiDAR のアダプティブプーリング
        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

        # ここでは in_channels=2 (positional encoding 用に2チャネル)
        # if self.config.use_point_pillars == True: ... (コメントアウト)
        # if self.config.use_target_point_image == True: in_channels += 1 (必要なら)

        in_channels = 2
        self.use_velocity = bool(use_velocity)  # True(1) なら速度を使う

        # 画像エンコーダ: グレースケール対応
        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)
        # LiDARエンコーダ: in_channels=2
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)

        # Transformer (GPT) の 4層 (マルチスケール)
        self.transformer1 = GPT(
            n_embd=self.image_encoder.features.feature_info[1]['num_chs'],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=self.use_velocity
        )
        self.transformer2 = GPT(
            n_embd=self.image_encoder.features.feature_info[2]['num_chs'],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=self.use_velocity
        )
        self.transformer3 = GPT(
            n_embd=self.image_encoder.features.feature_info[3]['num_chs'],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=self.use_velocity
        )
        self.transformer4 = GPT(
            n_embd=self.image_encoder.features.feature_info[4]['num_chs'],
            n_head=config.n_head,
            block_exp=config.block_exp,
            n_layer=config.n_layer,
            img_vert_anchors=config.img_vert_anchors,
            img_horz_anchors=config.img_horz_anchors,
            lidar_vert_anchors=config.lidar_vert_anchors,
            lidar_horz_anchors=config.lidar_horz_anchors,
            seq_len=config.seq_len,
            embd_pdrop=config.embd_pdrop,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            config=config,
            use_velocity=self.use_velocity
        )

        # チャネル数が一致しない場合のアダプタ
        if self.image_encoder.features.feature_info[4]['num_chs'] != self.config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                self.config.perception_output_features,
                kernel_size=(1,1)
            )
            self.change_channel_conv_lidar = nn.Conv2d(
                self.image_encoder.features.feature_info[4]['num_chs'],
                self.config.perception_output_features,
                kernel_size=(1,1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        # FPN (top-down) 処理
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor,
            mode='bilinear',
            align_corners=False
        )
        self.up_conv5 = nn.Conv2d(channel, channel, (1,1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1,1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1,1))
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1,1))
        self.fused_features = self.config.perception_output_features

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        """
        Args:
            image: グレースケール画像 (B,1,H,W)
            lidar: LiDAR (B,2,H_lidar,W_lidar) ← positional encodingなど使う
            velocity: ego-velocity (B,1) 
        """
        # グレースケール正規化
        if self.image_encoder.normalize:
            image_tensor = normalize_grayscale(image)
        else:
            image_tensor = image

        # LiDAR は positional encoding を書き込んでもよいが、ここでは minimal のみ:
        lidar_tensor = lidar

        # ========== Image Encoder ==========
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        image_features = self.image_encoder.features.layer1(image_features)

        # ========== LiDAR Encoder ==========
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)

        # (B, 64, H/4, W/4) 相当? => embed & Transformer
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)

        image_features_layer1 = F.interpolate(
            image_features_layer1,
            size=(image_features.shape[2], image_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        lidar_features_layer1 = F.interpolate(
            lidar_features_layer1,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        # ================= Layer2, Layer3, Layer4 同様に処理 =================
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)

        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(
            image_features_layer2,
            size=(image_features.shape[2],image_features.shape[3]),
            mode='bilinear', align_corners=False
        )
        lidar_features_layer2 = F.interpolate(
            lidar_features_layer2,
            size=(lidar_features.shape[2],lidar_features.shape[3]),
            mode='bilinear', align_corners=False
        )
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(
            image_features_layer3,
            size=(image_features.shape[2],image_features.shape[3]),
            mode='bilinear', align_corners=False
        )
        lidar_features_layer3 = F.interpolate(
            lidar_features_layer3,
            size=(lidar_features.shape[2],lidar_features.shape[3]),
            mode='bilinear', align_corners=False
        )
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features_layer4 = F.interpolate(
            image_features_layer4,
            size=(image_features.shape[2],image_features.shape[3]),
            mode='bilinear', align_corners=False
        )
        lidar_features_layer4 = F.interpolate(
            lidar_features_layer4,
            size=(lidar_features.shape[2],lidar_features.shape[3]),
            mode='bilinear', align_corners=False
        )
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

        # 512チャネルに揃える
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        # FPN 用
        # x4 = lidar_features

        # 途中で可視化などを行いたい場合: image_features_grid
        # image_features_grid = image_features

        # Global Pool for final fused features
        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        fused_features = image_features + lidar_features

        # FPN
        # features = self.top_down(x4)
        return fused_features
        # features, image_features_grid, 

class GPT(nn.Module):
    """
    The GPT block, now uses velocity by default (use_velocity=True).
    And is updated similarly to transfuser_stereo.py's GPT.
    """
    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 img_vert_anchors, img_horz_anchors,
                 lidar_vert_anchors, lidar_horz_anchors,
                 seq_len, 
                 embd_pdrop, attn_pdrop, resid_pdrop, config, use_velocity=1):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 1  # only support seq_len=1 for now
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.config = config

        # pos embedding
        self.pos_emb = nn.Parameter(torch.zeros(
            1,
            self.seq_len * img_vert_anchors * img_horz_anchors +
            self.seq_len * lidar_vert_anchors * lidar_horz_anchors,
            n_embd
        ))

        self.use_velocity = bool(use_velocity)
        if self.use_velocity:
            self.vel_emb = nn.Linear(self.seq_len, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor, velocity):
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]
        assert self.seq_len == 1

        # reshape from (B, C, H, W) -> tokens
        image_tensor = (image_tensor
            .view(bz, self.seq_len, -1, img_h, img_w)
            .permute(0,1,3,4,2)
            .contiguous()
            .view(bz, -1, self.n_embd))
        lidar_tensor = (lidar_tensor
            .view(bz, self.seq_len, -1, lidar_h, lidar_w)
            .permute(0,1,3,4,2)
            .contiguous()
            .view(bz, -1, self.n_embd))

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        if self.use_velocity:
            velocity_embeddings = self.vel_emb(velocity)  # (B, n_embd)
            x = self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)
        else:
            x = self.pos_emb + token_embeddings

        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        # revert shape
        x = x.view(bz, self.seq_len*(self.img_vert_anchors*self.img_horz_anchors + self.lidar_vert_anchors*self.lidar_horz_anchors), self.n_embd)

        image_tensor_out = x[:,:self.seq_len*self.img_vert_anchors*self.img_horz_anchors,:]
        image_tensor_out = (image_tensor_out
            .contiguous()
            .view(bz*self.seq_len, -1, img_h, img_w))

        lidar_tensor_out = x[:,self.seq_len*self.img_vert_anchors*self.img_horz_anchors:,:]
        lidar_tensor_out = (lidar_tensor_out
            .contiguous()
            .view(bz*self.seq_len, -1, lidar_h, lidar_w))

        return image_tensor_out, lidar_tensor_out

# Grayscale ImageCNN / LidarEncoder classes
# same approach as transfuser_stereo
class ImageCNN(nn.Module):
    """
    Grayscale-friendly Image CNN (like transfuser_stereo).
    """
    def __init__(self, architecture='resnet34', normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None

        # If it's regnet or convnext, do partial rename
        # For the sake of demonstration, minimal example:
        if architecture.startswith('regnet'):
            self.features.conv1 = self.features.stem.conv
            self.features.bn1  = self.features.stem.bn
            self.features.act1 = nn.Sequential()
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.s1
            self.features.layer2 = self.features.s2
            self.features.layer3 = self.features.s3
            self.features.layer4 = self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()

        # Convert to grayscale
        _tmp = self.features.conv1
        use_bias = (_tmp.bias is not None)
        self.features.conv1 = nn.Conv2d(
            in_channels=1,  # grayscale
            out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size,
            stride=_tmp.stride,
            padding=_tmp.padding,
            bias=use_bias
        )
        if _tmp.weight is not None:
            # average across RGB dimension => single channel
            self.features.conv1.weight.data = _tmp.weight.data.mean(dim=1, keepdim=True)
        if use_bias:
            self.features.conv1.bias.data = _tmp.bias.data

class LidarEncoder(nn.Module):
    """
    Lidar Encoder, set in_channels=2 by default. Similar to transfuser_stereo approach.
    """
    def __init__(self, architecture='resnet18', in_channels=2):
        super().__init__()
        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        # minimal example for e.g. regnet
        if architecture.startswith('regnet'):
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        # Force 2-ch conv1
        _tmp = self._model.conv1
        use_bias = (_tmp.bias is not None)
        self._model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size,
            stride=_tmp.stride,
            padding=_tmp.padding,
            bias=use_bias
        )
        if _tmp.weight is not None:
            self._model.conv1.weight.data = _tmp.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        if use_bias:
            self._model.conv1.bias.data = _tmp.bias.data

        # remove the old conv, or rename as needed
        if hasattr(self._model, 'stem'):
            # del self._model.stem  # optional if we want to remove unused param
            pass

        import torch
        torch.cuda.empty_cache()

        del _tmp


class SelfAttention(nn.Module):
    """ same as original """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = self.query(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = self.value(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ same as original """
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
