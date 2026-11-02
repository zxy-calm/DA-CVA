import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np
from numbers import Number
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AdaptiveScaleSelection(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        self.scale1 = nn.Conv2d(in_channels, in_channels, 1, padding=0)
        self.scale2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1)
        self.scale3 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.scale4 = nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 4),  
            nn.Softmax(dim=1)
        )

        self.hca_branch = HierarchicalChannelAttention(channels=in_channels, reduction=reduction)
        self.ssa_branch = MedicalSpatialAttention()
        self.ffed_branch= FrequencyFusionEncoderDecoder(channels=in_channels)

        self.weight_hca = nn.Parameter(torch.ones(1))
        self.weight_ssa = nn.Parameter(torch.ones(1))
        self.weight_ffed = nn.Parameter(torch.ones(1))


        self.record_branch_stats = False
        self.branch_stats = None

    def forward(self, x):
        b, c, h, w = x.shape

        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        f4 = self.scale4(x)

        weights = self.gap(x).view(b, c)
        weights = self.fc(weights).view(b, 4, 1, 1, 1)

        features = torch.stack([f1, f2, f3, f4], dim=1)  # [b, 4, c, h, w]
        weighted_features = features * weights
        multi_scale_out = weighted_features.sum(dim=1)
        #multi_scale_out = x

        out_hca = self.hca_branch(multi_scale_out)
        out_ssa = self.ssa_branch(multi_scale_out)
        out_ffed= self.ffed_branch(multi_scale_out)

        final_out = (self.weight_hca * out_hca) +(self.weight_ssa * out_ssa)+(self.weight_ffed * out_ffed)+multi_scale_out#+(self.weight_ssa * out_ssa)


        if self.record_branch_stats:
            def feature_norm(tensor):
                return tensor.detach().flatten(1).norm(p=2, dim=1).mean().item()

            self.branch_stats = {
                "multi_scale": feature_norm(multi_scale_out),
                "hca": feature_norm(out_hca),
                "ssa": feature_norm(out_ssa),
                "ffed": feature_norm(out_ffed),
                "weighted_hca": feature_norm(self.weight_hca * out_hca),
                "weighted_ssa": feature_norm(self.weight_ssa * out_ssa),
                "weighted_ffed": feature_norm(self.weight_ffed * out_ffed),
                "final": feature_norm(final_out),
            }

        #final_out = (1.4* out_hca) + (0.6 * out_ssa)+multi_scale_out
        return final_out


class MedicalSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1


        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 3, padding=1)
        )

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  

        sobel_x = F.conv2d(avg_out, torch.tensor([[[[1, 0, -1],
                                                    [2, 0, -2],
                                                    [1, 0, -1]]]],
                                                 dtype=torch.float32, device=x.device), padding=1)
        sobel_y = F.conv2d(avg_out, torch.tensor([[[[1, 2, 1],
                                                    [0, 0, 0],
                                                    [-1, -2, -1]]]],
                                                 dtype=torch.float32, device=x.device), padding=1)
        gradient = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)

        x_combined = torch.cat([gradient, max_out], dim=1)

        attention = self.conv(x_combined)
        attention = self.sigmoid(attention)

        boundary_attention = self.boundary_conv(attention)
        enhanced_attention = attention + 0.3 * boundary_attention

        return x * enhanced_attention

class HierarchicalChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16, levels=2):
        super().__init__()
        self.levels = levels
        self.channels = channels

        self.gap_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(pool_size)
            for pool_size in [1, 3]
        ])

        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels * (pool_size ** 2), channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels)
            )
            for pool_size in [1, 3]
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        channel_weights = []
        for i in range(self.levels):
            pooled = self.gap_layers[i](x)
            flattened = pooled.reshape(b, -1)
            weights = self.fc_layers[i](flattened).reshape(b, c, 1, 1)
            channel_weights.append(weights)

        combined_weights = sum(channel_weights) / self.levels
        weights = self.sigmoid(combined_weights)

        return x * weights


class FrequencyFusionEncoderDecoder(nn.Module):
    def __init__(self, channels, hidden_channels=None, reduction=4, freq_ratio=8, latent_dim=None):
        super().__init__()

        self.channels = channels
        self.concat_channels = channels * 2
        self.freq_ratio = freq_ratio

        hidden_channels = hidden_channels or self.concat_channels
        mid_channels = max(self.concat_channels // reduction, 16)
        latent_dim = latent_dim or self.concat_channels
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.concat_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(mid_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_channels, self.concat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.concat_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(self.concat_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        self.kl_weight = 1e-1
        self.latent_pool = nn.AdaptiveAvgPool2d(1)
        self.z_mu = nn.Linear(mid_channels, latent_dim)
        self.z_std = nn.Linear(mid_channels, latent_dim)
        self.ffed_mu_z = None
        self.ffed_std_z = None

     def split_frequency(self, x):
        # Haar小波分解：LL作为低频结构，LH/HL/HH合成为高频细节
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        x00 = x_pad[:, :, 0::2, 0::2]
        x01 = x_pad[:, :, 0::2, 1::2]
        x10 = x_pad[:, :, 1::2, 0::2]
        x11 = x_pad[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        high = torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + 1e-8)

        # 小波分解会让空间尺寸减半，这里上采样回原尺寸以保持后续模块接口不变
        low = F.interpolate(ll, size=(h, w), mode="bilinear", align_corners=False)
        high = F.interpolate(high, size=(h, w), mode="bilinear", align_corners=False)

        return low, high


    def forward(self, x, return_freq=False):
        low, high = self.split_frequency(x)

        # [B, C, H, W] + [B, C, H, W] -> [B, 2C, H, W]
        freq_feature = torch.cat([low, high], dim=1)

        encoded = self.encoder(freq_feature)
        latent_stat = self.latent_pool(encoded).flatten(1)
        self.ffed_mu_z = self.z_mu(latent_stat)
        self.ffed_std_z = F.softplus(self.z_std(latent_stat) - 5, beta=1) + 1e-8

        decoded_2c = self.decoder(encoded)

        reduced = self.channel_reduce(decoded_2c)

        out = self.residual_weight * reduced

        if return_freq:
            return out, {
                "low": low,
                "high": high,
                "freq_feature": freq_feature,
                "encoded": encoded,
                "decoded_2c": decoded_2c,
                "reduced": reduced
            }

        return out


class MyModels(nn.Module):
    def __init__(self, num_classes, num_experts=4,
                 pretrained_path="../pretrained/convnext_tiny-983f1562.pth"):
        super(MyModels, self).__init__()
        self.K = 128
        self.num_classes = num_classes
        self.ablate_experts = False
        self.ablate_backbone = False
        self.ablate_routing = set()
        convnext = models.convnext_tiny(weights=None)
        state_dict = (torch.load(pretrained_path, weights_only=True, map_location="cpu")
                      if pretrained_path is not None else None)
        # convnext = models.convnext_base(weights=None)
        # state_dict = torch.load("../pretrained/convnext_base-6075fbad.pth", weights_only=True)
        if state_dict is not None:
            load_convnext_state_dict(convnext, state_dict)

        features = [*convnext.features.children()]
        self.backbone = nn.Sequential(*features)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_output = self.backbone(dummy_input)
            backbone_out_channels = dummy_output.shape[1]

        self.projector = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(backbone_out_channels, self.K*2)
        )

        # 专家模型
        self.experts = nn.ModuleList()
        for _ in range(num_experts):

            expert = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=4),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(128, 384, kernel_size=7, stride=2, padding=1),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(384, 384, kernel_size=7, stride=3, padding=2, groups=384),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=1),
                nn.BatchNorm2d(384),
                nn.LeakyReLU(inplace=True),

                AdaptiveScaleSelection(in_channels=384),

                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),

                nn.Linear(384, 384),
                nn.LeakyReLU(),
                nn.Linear(384, self.K*2)  
            )
            self.experts.append(expert)

        # 路由网络（MLP）
        self.routing_network_emb = nn.Sequential(
            nn.Linear(self.K, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        self.routing_network_mu = nn.Sequential(
            nn.Linear(self.K, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        self.routing_network_std = nn.Sequential(
            nn.Linear(self.K, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )

        # 残差权重 α
        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = nn.Parameter(torch.tensor(0.8))
        self.gamme = nn.Parameter(torch.tensor(0.8))

        # 分类头
        self.classifier = nn.Linear(self.K, num_classes)
        self.class_prior_mu = nn.Parameter(torch.zeros(num_classes, self.K))
        self.class_prior_log_std = nn.Parameter(torch.zeros(num_classes, self.K))

        ffed_latent_dim = None
        for module in self.experts.modules():
            if isinstance(module, FrequencyFusionEncoderDecoder):
                ffed_latent_dim = module.latent_dim
                break
        self.ffed_prior_mu = nn.Parameter(torch.zeros(num_classes, ffed_latent_dim))
        self.ffed_prior_log_std = nn.Parameter(torch.zeros(num_classes, ffed_latent_dim))


    def forward(self, x):
        if self.ablate_backbone and self.ablate_experts:
            raise ValueError("ablate_backbone and ablate_experts cannot be enabled at the same time.")

        if self.ablate_backbone:
            expert_outputs = []
            expert_mus = []
            expert_stds = []
            for expert in self.experts:
                out = expert(x)
                mu_e = out[:, :self.K]
                std_e = F.softplus(out[:, self.K:] - 5, beta=1)
                emb_e = self.reparametrize_n(mu_e, std_e)
                expert_outputs.append(emb_e)
                expert_mus.append(mu_e)
                expert_stds.append(std_e)

            expert_outputs = torch.stack(expert_outputs, dim=1)
            expert_mus = torch.stack(expert_mus, dim=1)
            expert_stds = torch.stack(expert_stds, dim=1)

            emb_ex = expert_outputs.mean(dim=1)
            mu_ex = expert_mus.mean(dim=1)
            std_ex = expert_stds.mean(dim=1)
            logits = self.classifier(emb_ex)
            return logits, (mu_ex, std_ex)

        # main model 特征
        x_c = self.backbone(x).mean([-2, -1])
        x_c = self.projector(x_c)

        mu_c = x_c[:, :self.K]
        std_c = F.softplus(x_c[:, self.K:] - 5, beta=1)
        emb_c = self.reparametrize_n(mu_c, std_c)

        if self.ablate_experts:
            logits = self.classifier(emb_c)
            return logits, (mu_c, std_c)

        # 专家特征
        expert_outputs = []
        expert_mus = []
        expert_stds = []
        for expert in self.experts:
            out = expert(x)
            mu_e = out[:, :self.K]
            std_e = F.softplus(out[:, self.K:] - 5, beta=1)
            emb_e = self.reparametrize_n(mu_e, std_e)
            expert_outputs.append(emb_e)
            expert_mus.append(mu_e)
            expert_stds.append(std_e)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, D]
        expert_mus = torch.stack(expert_mus, dim=1)  # [B, num_experts, D]
        expert_stds = torch.stack(expert_stds, dim=1)  # [B, num_experts, D]

        # 路由消融：对应专家输出改为等权平均，其余路由保持不变。
        if "mu" in self.ablate_routing:
            mu_ex = expert_mus.mean(dim=1)
        else:
            routing_weights_mu = self.routing_network_mu(mu_c)
            routing_weights_mu = torch.softmax(routing_weights_mu, dim=1)
            mu_ex = torch.sum(routing_weights_mu.unsqueeze(-1) * expert_mus, dim=1)

        if "std" in self.ablate_routing:
            std_ex = expert_stds.mean(dim=1)
        else:
            routing_weights_std = self.routing_network_std(std_c)
            routing_weights_std = torch.softmax(routing_weights_std, dim=1)
            std_ex = torch.sum(routing_weights_std.unsqueeze(-1) * expert_stds, dim=1)

        if "emb" in self.ablate_routing:
            emb_ex = expert_outputs.mean(dim=1)
        else:
            routing_weights_emb = self.routing_network_emb(emb_c)
            routing_weights_emb = torch.softmax(routing_weights_emb, dim=1)
            emb_ex = torch.sum(routing_weights_emb.unsqueeze(-1) * expert_outputs, dim=1)

        # 残差连接
        x_mu = self.beta * mu_c + (1 - self.beta) * mu_ex
        x_std = self.gamme * std_c + (1 - self.gamme) * std_ex
        x_o = self.alpha * emb_c + (1 - self.alpha) * emb_ex

        # 分类
        logits = self.classifier(x_o)


        return logits, (x_mu, x_std)  # ,div_loss

     def reparametrize_n(self, mu, std, n=1):

        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_().to(mu.device))

        return mu + eps * std

    def get_ffed_prior(self, y):
        # 按类别标签取出FFED对应的类别条件先验
        prior_mu = self.ffed_prior_mu[y]
        prior_std = torch.exp(self.ffed_prior_log_std[y]) + 1e-8
        return prior_mu, prior_std

    def get_ffed_loss(self, y=None, use_class_prior_kl=True):
        # 汇总所有专家分支中的FFED辅助损失
        ffed_loss = 0.0
        for module in self.modules():
            if not isinstance(module, FrequencyFusionEncoderDecoder):
                continue
            if module.ffed_mu_z is None or module.ffed_std_z is None:
                continue

            mu_z = module.ffed_mu_z
            std_z = module.ffed_std_z
            if use_class_prior_kl and y is not None:
                prior_mu, prior_std = self.get_ffed_prior(y)
                ffed_kl_loss = 0.5 * (
                    2 * (torch.log(prior_std + 1e-8) - torch.log(std_z + 1e-8))
                    + ((mu_z - prior_mu) / (prior_std + 1e-8)) ** 2
                    + (std_z / (prior_std + 1e-8)) ** 2
                    - 1
                )
            else:
                prior_mu = torch.zeros_like(mu_z)
                prior_std = torch.ones_like(std_z)
                ffed_kl_loss = 0.5 * (
                    2 * (torch.log(prior_std + 1e-8) - torch.log(std_z + 1e-8))
                    + ((mu_z - prior_mu) / (prior_std + 1e-8)) ** 2
                    + (std_z / (prior_std + 1e-8)) ** 2
                    - 1
                )

            ffed_loss = ffed_loss + module.kl_weight * ffed_kl_loss.mean()
        return ffed_loss

    def get_encoding(self, x):
        if self.ablate_backbone and self.ablate_experts:
            raise ValueError("ablate_backbone and ablate_experts cannot be enabled at the same time.")

        if self.ablate_backbone:
            expert_outputs = []
            for expert in self.experts:
                out = expert(x)
                mu_e = out[:, :self.K]
                std_e = F.softplus(out[:, self.K:] - 5, beta=1)
                emb_e = self.reparametrize_n(mu_e, std_e)
                expert_outputs.append(emb_e)

            expert_outputs = torch.stack(expert_outputs, dim=1)
            return expert_outputs.mean(dim=1)
        # main model 特征
        x_c = self.backbone(x).mean([-2, -1])
        x_c = self.projector(x_c)

        mu_c = x_c[:, :self.K]
        std_c = F.softplus(x_c[:, self.K:] - 5, beta=1)
        emb_c = self.reparametrize_n(mu_c, std_c)

        if self.ablate_experts:
            return emb_c

        # 专家特征
        expert_outputs = []
        expert_mus = []
        expert_stds = []
        for expert in self.experts:
            out = expert(x)
            mu_e = out[:, :self.K]
            std_e = F.softplus(out[:, self.K:] - 5, beta=1)
            emb_e = self.reparametrize_n(mu_e, std_e)
            expert_outputs.append(emb_e)
            expert_mus.append(mu_e)
            expert_stds.append(std_e)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, D]

        if "emb" in self.ablate_routing:
            emb_ex = expert_outputs.mean(dim=1)
        else:
            routing_weights_emb = self.routing_network_emb(emb_c)
            routing_weights_emb = torch.softmax(routing_weights_emb, dim=1)
            emb_ex = torch.sum(routing_weights_emb.unsqueeze(-1) * expert_outputs, dim=1)

        # 残差连接
        x_o = self.alpha * emb_c + (1 - self.alpha) * emb_ex

        return x_o


if __name__ == "__main__":
    inputs = torch.zeros(1, 3, 256, 256)
    net = MyModels(num_classes=5)
    logits, (x_mu, x_std) = net(inputs)
    print(logits)
