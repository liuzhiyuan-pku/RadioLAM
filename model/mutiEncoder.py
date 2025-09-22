import torch
import torch.nn as nn
import torchvision.models as models
import math

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerEmbedding(nn.Module):
    def __init__(self, output_dim, input_dim=4, linear_dim = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, output_dim, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(output_dim),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.mlpEmbedding = nn.Sequential(
            nn.Linear(output_dim, linear_dim),
            nn.SiLU(),
            nn.Linear(linear_dim, output_dim),
        )


    def forward(self, x):
        xyz = x[..., :3]
        # print(xyz.shape)
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = x.permute(0, 2, 1)

        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = self.mlpEmbedding(x)

        return x


def value_embedding(heightsteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param heightsteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=heightsteps.device)
    args = heightsteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class MultiModelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.PointEmbeddingLength = config["PointEmbeddingLength"]
        self.BuildingEmbeddingLength = config["BuildingEmbeddingLength"]
        self.HeightEmbeddingLength = config["HeightEmbeddingLength"]
        self.FeqEmbeddingLength = config["FeqEmbeddingLength"]
        self.EnvEmbeddingLength = config["EnvEmbeddingLength"]

        self.env_classes_num = config["env_classes_num"]

        self.LinearDim = config["LinearDim"]

        self.PointEmbedding = PointTransformerEmbedding(output_dim=self.PointEmbeddingLength)

        self.BuildingEmbedding = models.resnet18(pretrained=False)
        self.BuildingEmbedding.fc = nn.Linear(512, self.BuildingEmbeddingLength)
        self.BuildingEmbedding.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.HeightEmbedding = nn.Sequential(
            nn.Linear(self.HeightEmbeddingLength, self.LinearDim),
            nn.SiLU(),
            nn.Linear(self.LinearDim, self.HeightEmbeddingLength),
        )

        self.FeqEmbedding = nn.Sequential(
            nn.Linear(self.FeqEmbeddingLength, self.LinearDim),
            nn.SiLU(),
            nn.Linear(self.LinearDim, self.FeqEmbeddingLength),
        )

        self.EnvEmbedding = nn.Sequential(
            nn.Embedding(self.env_classes_num, self.EnvEmbeddingLength),
            nn.Linear(self.EnvEmbeddingLength, self.LinearDim),
            nn.SiLU(),
            nn.Linear(self.LinearDim, self.EnvEmbeddingLength),
        )


    def forward(self, point_cloud, building, terrain, height, frequencies, env):
        point_embedding = self.PointEmbedding(point_cloud)
        building_embedding = self.BuildingEmbedding(torch.cat([building, terrain],dim=1))

        height_vector = value_embedding(height.view(-1), self.HeightEmbeddingLength)
        height_embedding = self.HeightEmbedding(height_vector)

        feq_vector = value_embedding(frequencies.view(-1), self.FeqEmbeddingLength)
        feq_embedding = self.FeqEmbedding(feq_vector)

        env_embedding = self.EnvEmbedding(env.view(-1).to(torch.long))

        return torch.stack([point_embedding, building_embedding, torch.cat([height_embedding, feq_embedding, env_embedding],dim=1)], dim=1)


