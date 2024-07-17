import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from models.transformer import PositionalEncoding, TransformerEncoderLayer, TransformerEncoder
from models.encoder import Content_TR
from einops import rearrange, repeat
import logging
from z_new_start.FontTransformer import TransformerDecoderLayer, TransformerDecoder
from torch import nn, Tensor
import torch

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FontModel(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_head=8,
                 num_encoder_layers=2,
                 num_glyph_encoder_layers=2,
                 num_gly_decoder_layers=2,
                 dim_feedforward=2048,  # 前馈神经网络中隐藏层的大小
                 dropout=0.2,
                 activation="relu",
                 # activation = "gelu",
                 normalize_before=True,  # 应用多头注意力和前馈神经网络之前是否对输入进行层归一化
                 return_intermediate_dec=True,  # 是否在解码过程中返回中间结果
                 train_conf=None
                 ):
        super(FontModel, self).__init__()
        self.train_conf = train_conf

        # encoder
        self.feat_encoder = self._build_feature_encoder()
        self.encoder_layer = TransformerEncoderLayer(
            d_model, num_head, dim_feedforward, dropout, activation, normalize_before
        )
        # self.base_encoder = self._build_base_encoder(
        #     num_encoder_layers
        # )
        self.glyph_encoder = self._build_glyph_encoder(
            d_model, normalize_before, num_glyph_encoder_layers
        )
        self.font_encoder = self._build_font_encoder(
            d_model, normalize_before, num_glyph_encoder_layers
        )
        self.content_encoder = Content_TR(d_model=d_model, num_encoder_layers=num_encoder_layers)

        # decoder
        self.decoder_layer = TransformerDecoderLayer(
            d_model, num_head, dim_feedforward, dropout, activation, normalize_before
        )
        self.glyph_transformer_decoder = self._build_glyph_decoder(
            d_model, normalize_before, num_gly_decoder_layers, return_intermediate_dec
        )
        self.font_transformer_decoder = self._build_font_decoder(
            d_model, normalize_before, num_gly_decoder_layers, return_intermediate_dec
        )

        # other
        # self.pro_mlp_font = nn.Sequential(
        #     nn.Linear(512, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 256)
        # )
        # self.pro_mlp_glyph = nn.Sequential(
        #     nn.Linear(512, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 256)
        # )

        self.SeqtoEmb = Seq2Emb(output_dim=d_model)
        self.EmbtoSeq = Emb2Seq(input_dim=d_model)
        self.add_position = PositionalEncoding(dim=d_model, dropout=0.1)
        # 参数重置 用于初始化模型的参数
        self._init_parameters()

    def _build_feature_encoder(self):
        # 图像的特征提取卷积层
        # 此处使用一个卷积层和一个预训练的 ResNet-18 模型的特征提取器
        # *() 将列表中的元素作为多个参数传递给 nn.Sequential，而不是将整个列表作为一个参数
        # Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:/Users/liuch/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
        return nn.Sequential(*(  # 一个输入通道，输出64个通道。卷积核大小为7，步长为2，填充为3。bias=False不使用偏置项
                [nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)]
                +
                # 获取了 ResNet-18 模型的子模块列表，然后去掉了列表的第一个和最后两个模块。
                # 这些被去掉的模块通常是 ResNet-18 模型的头部，包括全局平均池化层和全连接层。
                list(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[1:-2]
        ))

    # Transformer 编码器
    def _build_base_encoder(self, num_encoder_layers):
        return TransformerEncoder(self.encoder_layer, num_encoder_layers)

    def _build_glyph_encoder(self, d_model, normalize_before, num_glyph_encoder_layers):
        glyph_norm = nn.LayerNorm(d_model) if normalize_before else None
        return TransformerEncoder(self.encoder_layer, num_glyph_encoder_layers, glyph_norm)

    def _build_font_encoder(self, d_model, normalize_before, num_glyph_encoder_layers):
        font_norm = nn.LayerNorm(d_model) if normalize_before else None
        return TransformerEncoder(self.encoder_layer, num_glyph_encoder_layers, font_norm)

    def _build_glyph_decoder(self, d_model, normalize_before, num_gly_decoder_layers, return_intermediate_dec):
        gly_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        return TransformerDecoder(self.decoder_layer, num_gly_decoder_layers, gly_decoder_norm,
                                  return_intermediate_dec)

    def _build_font_decoder(self, d_model, normalize_before, num_gly_decoder_layers, return_intermediate_dec):
        font_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        return TransformerDecoder(self.decoder_layer, num_gly_decoder_layers, font_decoder_norm,
                                  return_intermediate_dec)

    def _init_parameters(self):
        for p in self.parameters():
            """
            如果参数的维度大于 1，则使用 Xavier 均匀初始化方法来初始化参数 p。
            Xavier 初始化是一种常用的权重初始化方法，它确保前向传播和反向传播的信号在深度神经网络中保持平衡，
            有助于加快训练速度和提高模型的性能。
            总的来说，这段代码的作用是在模型中对所有二维参数（即权重矩阵）
            使用 Xavier 均匀初始化方法进行初始化。这是一种常见的初始化策略，特别适用于深度学习模型。
            """
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.kaiming_uniform_(p, a=0, mode='fan_in', nonlinearity='relu')

    def _random_double_sampling(self, x, ratio=0.25):
        L, B, N, D = x.shape  # length, batch, group_number, dim
        x = rearrange(x, "L B N D -> B N L D")
        # 这个噪声张量用于对每个组内的序列进行随机排序。 noise in [0, 1]
        noise = torch.rand(B, N, L, device=x.device)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)

        # 计算出需要保留的锚点（anchor）和正样本（positive）的数量。
        # 锚点数量是总长度的ratio倍，正样本数量是锚点数量的2倍。
        anchor_tokens, pos_tokens = int(L * ratio), int(L * 2 * ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos

    def forward(self, same_style_img_list, std_coors, char_img_gt):
        check_tensor(same_style_img_list, "same_style_img_list")
        check_tensor(std_coors, "std_coors")
        check_tensor(char_img_gt, "char_img_gt")
        logger.debug(
            f"Input shapes: \n"
            f"same_style_img_list={same_style_img_list.shape}\n"
            f"std_coors={std_coors.shape}\n"
            f"char_img_gt={char_img_gt.shape}"
        )
        # [bs, num_img, C, 64, 64] == [B,N,C,H,W]
        batch_size, num_img, in_planes, h, w = same_style_img_list.shape
        # [B,N,C,H,W]==>[B*N,C,h,w]
        style_img_list = same_style_img_list.view(-1, in_planes, h, w)
        logger.debug(f"style_img_list shape: {style_img_list.shape}")

        # [a] 编码风格图像特征
        # [B*N,C,h,w]==>[B*N, 64, h/2, w/2]==> [B*N, 512, h/32, w/32]
        feat = self.feat_encoder(style_img_list)
        logger.debug(f"feat shape after feat_encoder: {feat.shape}")
        # [B*N, 512, h/32, w/32]==>[B*N, 512, h/32 * w/32] ==> [h/32*w/32,B*N,512] = [4, 16, 512]
        feat = feat.view(batch_size * num_img, 512, -1).permute(2, 0, 1)
        logger.debug(f"feat shape after view and permute: {feat.shape}")
        feat = self.add_position(feat)
        logger.debug(f"feat shape after add_position: {feat.shape}")
        # feat = self.base_encoder(feat)
        # logger.info(f"feat shape after base_encoder: {feat.shape}")

        # [b] 处理和重组特征
        glyph_feat = self.glyph_encoder(feat)
        check_tensor(glyph_feat, "glyph_feat after glyph_encoder")
        font_feat = self.font_encoder(feat)
        check_tensor(font_feat, "font_feat after font_encoder")
        # [h/32*w/32,B*N,512] ==> [h/32*w/32,2*B,N/2,512]
        glyph_memory = rearrange(glyph_feat, 't (b p n) c -> t (p b) n c',
                                 b=batch_size, p=2, n=num_img // 2)
        font_memory = rearrange(font_feat, 't (b p n) c -> t (p b) n c',
                                b=batch_size, p=2, n=num_img // 2)  # [4, 2*B, N, C]
        logger.debug(f"glyph_memory shape: {glyph_memory.shape} font_memory shape: {font_memory.shape}")
        memory_feat = rearrange(font_memory, 't b n c ->(t n) b c')  # 特征重组

        # # [c] 计算对比学习嵌入（NCE）特征
        # # 计算用于对比学习的嵌入特征，分别处理 font 和 glyph 特征
        # # [c.1] font-NCE
        # compact_feat = torch.mean(memory_feat, 0)  # 特征平均
        # logger.debug(f"compact_feat shape: {compact_feat.shape}")
        # pro_emb = self.pro_mlp_font(compact_feat)  # 特征投影
        # query_emb = pro_emb[:batch_size, :]
        # pos_emb = pro_emb[batch_size:, :]  # 特征拆分
        # logger.debug(f"query_emb shape: {query_emb.shape} pos_emb shape: {pos_emb.shape}")
        # nce_emb = torch.stack((query_emb, pos_emb), 1)
        # nce_emb = nn.functional.normalize(nce_emb, p=2, dim=2)  # 特征组合和归一化
        # logger.debug(f"nce_emb shape: {nce_emb.shape}")
        #
        # # [c.2] glyph-NCE
        # patch_emb = glyph_memory[:, :batch_size]  # 特征采样
        # logger.debug(f"patch_emb shape: {patch_emb.shape}")
        # anc, positive = self._random_double_sampling(patch_emb)
        # anc = anc.reshape(batch_size, -1, anc.shape[-1])
        # anc_compact = torch.mean(anc, 1, keepdim=True)
        # anc_compact = self.pro_mlp_glyph(anc_compact)  # 特征重组和投影
        # positive = positive.reshape(batch_size, -1, positive.shape[-1])
        # positive_compact = torch.mean(positive, 1, keepdim=True)
        # positive_compact = self.pro_mlp_glyph(positive_compact)
        # logger.debug(f"anc_compact shape: {anc_compact.shape} positive_compact shape: {positive_compact.shape}")
        # nce_emb_patch = torch.cat((anc_compact, positive_compact), 1)
        # nce_emb_patch = nn.functional.normalize(nce_emb_patch, p=2, dim=2)  # 特征组合和归一化
        # logger.debug(f"nce_emb_patch shape: {nce_emb_patch.shape}")

        # [d] 解码和生成预测序列
        font_style = memory_feat[:, :batch_size, :]
        glyph_style = glyph_memory[:, :batch_size]
        logger.debug(f"glyph_style shape:{glyph_style.shape} font_style shape:{font_style.shape}")
        # [h/32*w/32,2*B,N/2,512] ==> [h/32*w/32,B,N/2,512]
        glyph_style = rearrange(glyph_style, 't b n c -> (t n) b c')
        logger.debug(f"glyph_style after rearrange shape: {glyph_style.shape}")
        # 处理标准坐标
        # [B,20,200,4] ==> [B,4000,4]==> [4000,B,4]
        std_coors = rearrange(std_coors, 'b t n c -> b (t n) c').permute(1, 0, 2)
        logger.debug(f"std_coors shape: {std_coors.shape}")
        # [4000,B,4]==>[4000, B, 512]
        seq_emb = self.SeqtoEmb(std_coors)
        T, B, C = seq_emb.shape
        logger.debug(f"seq_emb shape: {seq_emb.shape}")
        check_tensor(seq_emb, "seq_emb after SeqtoEmb")

        # 提取目标字符图像的内容特征
        # [bs, 1, 64, 64] = [B,C,H,W]==> [B,512,H/32,W/32]==>rearrange(x,'n c h w -> (h w) n c')=[4, B, 512]
        char_emb = self.content_encoder(char_img_gt)
        logger.debug(f"char_emb shape: {char_emb.shape}")
        check_tensor(char_emb, "char_emb after content_encoder")
        # [4, B, 512]==>[B, 512]
        char_emb = torch.mean(char_emb, 0)
        # ==>[1, B, 512]
        char_emb = repeat(char_emb, 'n c -> t n c', t=1)

        # [4000,B,512] + [1, B, 512] = [4001, B, 512]
        tgt = torch.cat((char_emb, seq_emb), 0)
        logger.debug(f"tgt shape: {tgt.shape}")
        tgt_mask = generate_square_subsequent_mask(T + 1).to(tgt)
        # check_tensor(tgt_mask, "tgt_mask")
        tgt = self.add_position(tgt)
        logger.debug(f"tgt shape after add_position: {tgt.shape}")
        check_tensor(tgt, "tgt after add_position")

        # [e] 使用解码器生成预测序列
        # [1, 4004, 8, 512]
        font_hs = self.font_transformer_decoder(tgt, font_style, tgt_mask=tgt_mask)
        hs = self.glyph_transformer_decoder(font_hs[-1], glyph_style, tgt_mask=tgt_mask)
        logger.debug(f"font_hs shape: {font_hs.shape} hs shape: {hs.shape}")
        check_tensor(font_hs, "font_hs after font_transformer_decoder")
        check_tensor(hs, "hs after glyph_transformer_decoder")
        # [4004, 8, 512]
        h = hs.transpose(1, 2)[-1]
        logger.debug(f"h shape: {h.shape}")
        check_tensor(h, "h after transpose")

        # [f] Generating prediction sequence using EmbtoSeq network
        pred_sequence = self.EmbtoSeq(h)
        logger.debug(f"pred_sequence shape: {pred_sequence.shape}")
        check_tensor(pred_sequence, "pred_sequence after EmbtoSeq")

        pred_sequence = pred_sequence[:, :T, :]
        logger.debug(f"pred_sequence shape2: {pred_sequence.shape}")
        pred_sequence = pred_sequence.view(
            B, self.train_conf['max_stroke'], self.train_conf['max_per_stroke_point'], -1
        )
        logger.debug(f"pred_sequence after view: {pred_sequence.shape}")
        check_tensor(pred_sequence, "pred_sequence after view")
        return pred_sequence

    @torch.jit.export
    def inference(self, img_list):
        self.eval()
        device = next(self.parameters()).device
        outputs = []
        with torch.no_grad():
            for img in img_list:
                img = img.unsqueeze(0).to(device)
                std_coors = torch.zeros(
                    (1, self.train_conf['max_stroke'], self.train_conf['max_per_stroke_point'], 4)).to(device)
                char_img_gt = img
                pred_sequence = self.forward(img, std_coors, char_img_gt)
                outputs.append(pred_sequence.cpu().numpy())
        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    torch.triu(torch.ones(sz, sz))：生成一个大小为 (sz, sz) 的上三角矩阵，
    其中上三角部分为 1，其余部分为 0。
    (torch.triu(torch.ones(sz, sz)) == 1)：将上三角矩阵转换为布尔张量，
    上三角部分为 True，其余部分为 False。
    .transpose(0, 1)：将布尔张量转置，得到一个下三角矩阵，
    其中下三角部分为 True，其余部分为 False。
    .float()：将布尔张量转换为浮点数张量。
    .masked_fill(mask == 0, float('-inf'))：将布尔张量中值为 False（即 0）的位置填充为 float('-inf')。
    .masked_fill(mask == 1, float(0.0))：将布尔张量中值为 True（即 1）的位置填充为 float(0.0)。
    i.e.
    [[0, inf, inf],
    [0, 0, inf],
    [0, 0, 0]].
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask
        .float()
        .masked_fill(mask == 0, float('-inf'))
        # .masked_fill(mask == 0, -1e9)
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Seq2Emb(nn.Module):

    def __init__(self, output_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(4, 256)
        self.fc_2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x


class Emb2Seq(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 256)
        self.fc_2 = nn.Linear(256, 4)  # 4 * 24 + 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x


def check_tensor(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.error(f"{name} contains NaN or Inf values")
