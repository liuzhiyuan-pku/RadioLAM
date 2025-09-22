import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


def create_norm(type, shape):
    if type == 'ln':
        return nn.LayerNorm(shape)
    elif type == 'gn':
        return nn.GroupNorm(32, shape[0])


def create_activation(type):
    if type == 'relu':
        return nn.ReLU()
    elif type == 'silu':
        return nn.SiLU()


from einops import rearrange


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1)

    def forward(self, x, context):
        """
        Args:
            x: [batch_size, c, h, w]
            context: [batch_size, emb_dim]
        Returns:
            out: [batch_size, c, h, w]
        """
        b, c, h, w = x.shape

        # 将特征向量扩展为序列形式
        context = context.unsqueeze(1)  # [b, 1, emb_dim]

        # 投影输入特征
        x_in = self.proj_in(x)  # [b, emb_dim, h, w]
        x_in = rearrange(x_in, 'b c h w -> b (h w) c')  # [b, h*w, emb_dim]

        # 计算Q, K, V
        Q = self.Wq(x_in)  # [b, h*w, emb_dim]
        K = self.Wk(context)  # [b, 1, emb_dim]
        V = self.Wv(context)  # [b, 1, emb_dim]

        # 计算注意力权重
        att_weights = torch.einsum('bid,bjd->bij', Q, K) * self.scale  # [b, h*w, 1]
        att_weights = F.softmax(att_weights, dim=-1)

        # 应用注意力
        out = torch.einsum('bij,bjd->bid', att_weights, V)  # [b, h*w, emb_dim]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [b, emb_dim, h, w]
        out = self.proj_out(out)  # [b, c, h, w]

        return out + x  # 残差连接



class ResBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, time_c, norm_type='ln', activation_type='silu'):
        super().__init__()
        self.norm1 = create_norm(norm_type, shape)
        self.norm2 = create_norm(norm_type, (out_c, *shape[1:]))
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.time_layer = nn.Linear(time_c, out_c)
        self.activation = create_activation(activation_type)
        if in_c == out_c:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x, t):
        n = t.shape[0]
        out = self.activation(self.norm1(x))
        out = self.conv1(out)

        t = self.activation(t)
        t = self.time_layer(t).reshape(n, -1, 1, 1)
        out = out + t

        out = self.activation(self.norm2(out))
        out = self.conv2(out)
        out += self.residual_conv(x)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, shape, dim, norm_type='ln'):
        super().__init__()
        self.norm = create_norm(norm_type, shape)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        norm_x = self.norm(x)
        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)

        q = q.reshape(n, c, h * w).permute(0, 2, 1)
        k = k.reshape(n, c, h * w)
        qk = torch.bmm(q, k) / c ** 0.5
        qk = torch.softmax(qk, -1)
        qk = qk.permute(0, 2, 1)
        v = v.reshape(n, c, h * w)
        res = torch.bmm(v, qk).reshape(n, c, h, w)
        res = self.out(res)
        return x + res


class ResAttnBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, time_c, with_attn, norm_type='ln', activation_type='silu'):
        super().__init__()
        self.res_block = ResBlock(shape, in_c, out_c, time_c, norm_type, activation_type)
        if with_attn:
            self.attn_block = SelfAttentionBlock((out_c, shape[1], shape[2]), out_c, norm_type)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        x = self.res_block(x, t)
        x = self.attn_block(x)
        return x


class ResAttnBlockMid(nn.Module):
    def __init__(self, shape, in_c, out_c, time_c, with_attn, norm_type='ln', activation_type='silu'):
        super().__init__()
        self.res_block1 = ResBlock(shape, in_c, out_c, time_c, norm_type, activation_type)
        self.res_block2 = ResBlock((out_c, shape[1], shape[2]), out_c, out_c, time_c, norm_type, activation_type)
        if with_attn:
            self.attn_block = SelfAttentionBlock((out_c, shape[1], shape[2]), out_c, norm_type)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        x = self.res_block1(x, t)
        x = self.attn_block(x)
        x = self.res_block2(x, t)
        return x


class UNet_with_cond(nn.Module):
    def __init__(self, n_steps, img_shape, feature_dim=0, channels=[10, 20, 40, 80],
                 pe_dim=10, with_attns=False, norm_type='ln', activation_type='silu'):
        super().__init__()
        C, H, W = img_shape
        in_channels = C
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        self.NUM_RES_BLOCK = 2
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)
        if isinstance(with_attns, bool):
            with_attns = [with_attns] * layers

        self.pe = PositionalEncoding(n_steps, pe_dim)
        time_c = 4 * channels[0]
        self.pe_linears = nn.Sequential(
            nn.Linear(pe_dim, time_c),
            create_activation(activation_type),
            nn.Linear(time_c, time_c)
        )

        # Feature processing layers
        self.feature_dim = feature_dim
        if feature_dim > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, time_c),
                create_activation(activation_type),
                nn.Linear(time_c, time_c)
            )
            # 创建交叉注意力层
            self.cross_attns = nn.ModuleList([
                CrossAttentionBlock(channel, time_c)
                for channel in channels[:-1]
            ])
            # 中间层的交叉注意力
            self.mid_cross_attn = CrossAttentionBlock(channels[-1], time_c)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev_channel = channels[0]
        for channel, cH, cW, with_attn in zip(channels[0:-1], Hs[0:-1], Ws[0:-1], with_attns[0:-1]):
            encoder_layer = nn.ModuleList()
            for index in range(self.NUM_RES_BLOCK):
                if index == 0:
                    modules = ResAttnBlock(
                        (prev_channel, cH, cW), prev_channel, channel, time_c,
                        with_attn, norm_type, activation_type)
                else:
                    modules = ResAttnBlock((channel, cH, cW), channel, channel,
                                           time_c, with_attn, norm_type, activation_type)
                encoder_layer.append(modules)
            self.encoders.append(encoder_layer)

            # Add cross attention after each encoder block if feature_dim > 0
            if feature_dim > 0:
                self.cross_attns.append(CrossAttentionBlock(channel, time_c))

            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        cH = Hs[-1]
        cW = Ws[-1]
        channel = channels[-1]
        self.mid = ResAttnBlockMid((prev_channel, cH, cW), prev_channel,
                                   channel, time_c, with_attns[-1], norm_type,
                                   activation_type)

        prev_channel = channel
        for channel, cH, cW, with_attn in zip(channels[-2::-1], Hs[-2::-1],
                                              Ws[-2::-1], with_attns[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))

            decoder_layer = nn.ModuleList()
            for _ in range(self.NUM_RES_BLOCK):
                modules = ResAttnBlock((2 * channel, cH, cW), 2 * channel,
                                       channel, time_c, with_attn, norm_type,
                                       activation_type)
                decoder_layer.append(modules)

            self.decoders.append(decoder_layer)
            prev_channel = channel

        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, 1, 1)
        self.conv_out = nn.Conv2d(prev_channel, 1, 3, 1, 1)
        self.activation = create_activation(activation_type)

    def forward(self, x, t, features=None):
        t = self.pe(t)
        pe = self.pe_linears(t)

        # Process features if provided
        if self.feature_dim > 0 and features is not None:
            features = self.feature_proj(features)  # [b, time_c]
        else:
            features = torch.zeros_like(pe)  # dummy features if not provided

        x = self.conv_in(x)

        encoder_outs = []
        cross_attn_idx = 0
        for encoder, down in zip(self.encoders, self.downs):
            tmp_outs = []
            for index in range(self.NUM_RES_BLOCK):
                x = encoder[index](x, pe)
                tmp_outs.append(x)

            # 在编码器块后应用交叉注意力
            if self.feature_dim > 0:
                x = self.cross_attns[cross_attn_idx](x, features)
                cross_attn_idx += 1

            tmp_outs = list(reversed(tmp_outs))
            encoder_outs.append(tmp_outs)
            x = down(x)

        x = self.mid(x, pe)

        # 在中间层应用交叉注意力
        if self.feature_dim > 0:
            x = self.mid_cross_attn(x, features)

        for decoder, up, encoder_out in zip(self.decoders, self.ups, encoder_outs[::-1]):
            x = up(x)

            pad_x = encoder_out[0].shape[2] - x.shape[2]
            pad_y = encoder_out[0].shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))

            for index in range(self.NUM_RES_BLOCK):
                c_encoder_out = encoder_out[index]
                x = torch.cat((c_encoder_out, x), dim=1)
                x = decoder[index](x, pe)

        x = self.conv_out(self.activation(x))
        return x