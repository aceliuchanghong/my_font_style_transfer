import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from models.transformer import *
from models.encoder import Content_TR
from einops import rearrange
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FontModel(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_head=8,
                 num_encoder_layers=2,
                 num_glyph_encoder_layers=1,
                 num_gly_decoder_layers=2,
                 dim_feedforward=2048,  # 前馈神经网络中隐藏层的大小
                 dropout=0.2,
                 activation="relu",
                 normalize_before=True,  # 应用多头注意力和前馈神经网络之前是否对输入进行层归一化
                 return_intermediate_dec=True,  # 是否在解码过程中返回中间结果
                 train_conf=None
                 ):
        super(FontModel, self).__init__()
        self.train_conf = train_conf

        # 图像的特征提取卷积层
        # 此处使用一个卷积层和一个预训练的 ResNet-18 模型的特征提取器
        # *() 将列表中的元素作为多个参数传递给 nn.Sequential，而不是将整个列表作为一个参数
        # Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:/Users/liuch/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
        self.feat_encoder = nn.Sequential(*(  # 一个输入通道，输出64个通道。卷积核大小为7，步长为2，填充为3。bias=False不使用偏置项
                [nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)]
                +
                # 获取了 ResNet-18 模型的子模块列表，然后去掉了列表的第一个和最后两个模块。
                # 这些被去掉的模块通常是 ResNet-18 模型的头部，包括全局平均池化层和全连接层。
                list(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[1:-2]
        ))

        # Transformer 编码器
        encoder_layer = TransformerEncoderLayer(
            d_model, num_head, dim_feedforward, dropout, activation, normalize_before
        )
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # 风格特征编码器
        glyph_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.glyph_encoder = TransformerEncoder(
            encoder_layer, num_glyph_encoder_layers, glyph_norm
        )

        # 内容编码器 用于对输入的内容进行编码，以提取内容信息。
        self.content_encoder = Content_TR(
            d_model=d_model, num_encoder_layers=num_encoder_layers
        )

        # 风格特征解码器
        glyph_decoder_layers = TransformerDecoderLayer(
            d_model, num_head, dim_feedforward, dropout, activation
        )
        self.glyph_transformer_decoder = TransformerDecoder(
            glyph_decoder_layers, num_gly_decoder_layers
        )

        # 多层感知器（MLP，Multi-Layer Perceptron)
        # Gaussian Error Linear Unit
        self.pro_mlp_character = nn.Sequential(
            nn.Linear(512, 4096),
            nn.GELU(),
            nn.Linear(4096, 256)
        )

        """
        添加笔画宽度和颜色装饰网络
        width = self.stroke_width_network(features)
        color = self.color_network(features)
        由于 pro_mlp_writer 和 pro_mlp_character 生成的特征是 256 维度的，
        因此 stroke_width_network 和 color_network 接收 256 维度的输入。
        如果输入维度不同，需要根据实际情况进行调整。
        """
        self.stroke_width_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出一个值表示笔画宽度
        )
        self.color_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出三个值表示颜色的RGB通道
        )

        # 序列到emb (SeqtoEmb) 和 emb到序列 (EmbtoSeq)
        # 这两个模块用于处理序列数据和嵌入数据之间的转换。
        self.SeqtoEmb = Seq2Emb(output_dim=d_model)
        self.EmbtoSeq = Emb2Seq(input_dim=d_model)
        self.add_position = PositionalEncoding(dim=d_model, dropout=0.1)
        # 参数重置 用于初始化模型的参数
        self._init_parameters()

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

    def forward(self, same_style_img_list, std_coors, char_img_gt):
        logger.info(
            f"Input shapes: \n"
            f"same_style_img_list={same_style_img_list.shape}\n"
            f"std_coors={std_coors.shape}\n"
            f"char_img_gt={char_img_gt.shape}"
        )
        # [bs, num_img, C, 64, 64] == [B,N,C,H,W]
        batch_size, num_img, temp, h, w = same_style_img_list.shape
        # [B,N,C,H,W]==>[B*N,C,h,w]
        style_img_list = same_style_img_list.view(-1, temp, h, w)
        logger.info(f"style_img_list shape: {style_img_list.shape}")

        # 提取风格图像特征
        # [B*N,C,h,w]==>[B*N, 64, h/2, w/2]==> [B*N, 512, h/32, w/32]
        feat = self.feat_encoder(style_img_list)
        logger.info(f"feat shape after feat_encoder: {feat.shape}")
        # [B*N, 512, h/32, w/32]==>[B*N, 512, h/32 * w/32] ==> [h/32*w/32,B*N,512] = [4, 16, 512]
        feat = feat.view(batch_size * num_img, 512, -1).permute(2, 0, 1)
        logger.info(f"feat shape after view and permute: {feat.shape}")
        feat = self.add_position(feat)
        feat = self.base_encoder(feat)
        feat = self.glyph_encoder(feat)

        # 重新排列特征以分离风格和内容
        # [h/32*w/32,B*N,512] ==> [h/32*w/32,2*B,N/2,512]
        glyph_memory = rearrange(feat, 't (b p n) c -> t (p b) n c',
                                 b=batch_size, p=2, n=num_img // 2)
        logger.info(f"glyph_memory shape: {glyph_memory.shape}")
        # [h/32*w/32,2*B,N/2,512] ==> [h/32*w/32,B,N/2,512]
        glyph_style = glyph_memory[:, :batch_size]
        logger.info(f"glyph_style shape: {glyph_style.shape}")
        # [h/32*w/32,B,N/2,512] ==> [h/32*w/32*N/2,B,512]
        glyph_style = rearrange(glyph_style, 't b n c -> (t n) b c')
        logger.info(f"glyph_style shape after rearrange: {glyph_style.shape}")

        # 处理标准坐标
        # [8, 20, 200, 4]=[B,20,200,4] ==> [B,4000,4]
        std_coors = rearrange(std_coors, 'b t n c -> b (t n) c')
        logger.info(f"std_coors shape after rearrange: {std_coors.shape}")
        # [B,4000,4]==>[B,4000,512]==>[4000,B,512]
        seq_emb = self.SeqtoEmb(std_coors).permute(1, 0, 2)
        logger.info(f"seq_emb shape: {seq_emb.shape}")

        # 提取目标字符图像的内容特征
        # [bs, 1, 64, 64] = [B,C,H,W]==> [B,512,H/32,W/32]==>rearrange(x,'n c h w -> (h w) n c')=[4, B, 512]
        char_emb = self.content_encoder(char_img_gt)
        logger.info(f"char_emb shape: {char_emb.shape}")

        # 准备解码器输入
        # [4000,B,512] + [4, B, 512] = [4004, B, 512]
        tgt = torch.cat((char_emb, seq_emb), 0)
        logger.info(f"tgt shape: {tgt.shape}")
        T, N, C = tgt.shape
        tgt_mask = generate_square_subsequent_mask(sz=(T)).to(tgt.device)
        tgt = self.add_position(tgt)
        logger.info(f"tgt shape after add_position: {tgt.shape}")

        # 使用解码器生成预测序列
        # [1, 4004, 8, 512]
        hs = self.glyph_transformer_decoder(tgt, glyph_style, tgt_mask=tgt_mask)
        logger.info(f"hs shape: {hs.shape}")
        # [4004, 8, 512]
        h = hs.transpose(1, 2)[-1]
        logger.info(f"h shape: {h.shape}")
        pred_sequence = self.EmbtoSeq(h)
        logger.info(f"pred_sequence shape: {pred_sequence.shape}")

        B, T, _ = std_coors.shape # [B,4000,4]

        pred_sequence = pred_sequence[:, :T, :].view(B, self.train_conf['max_stroke'],
                                                     self.train_conf['max_per_stroke_point'], -1)
        logger.info(f"pred_sequence shape after view: {pred_sequence.shape}")
        return pred_sequence

    def inference(self, img_list):
        self.eval()  # 切换到评估模式
        device = next(self.parameters()).device
        outputs = []

        with torch.no_grad():  # 禁用梯度计算以提高推理速度并节省内存
            for img in img_list:
                img = img.unsqueeze(0).to(device)  # 增加批次维度并移动到设备上

                # 假设 std_coors 和 char_img_gt 是推理过程中需要的其他输入
                std_coors = torch.zeros((1, self.max_stroke, self.max_per_stroke_point, 4)).to(device)
                char_img_gt = img  # 在推理过程中，char_img_gt 可以是输入图像本身

                # 调用模型的 forward 方法进行推理
                pred_sequence = self.forward(img, std_coors, char_img_gt)

                # 将预测结果添加到输出列表中
                outputs.append(pred_sequence.cpu().numpy())  # 转换为 numpy 数组以便后续处理

        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    sz：表示掩码的大小，即掩码的行数和列数。
    Tensor：返回一个 PyTorch 张量。

    torch.triu(torch.ones(sz, sz))：生成一个大小为 (sz, sz) 的上三角矩阵，
    其中上三角部分为 1，其余部分为 0。
    (torch.triu(torch.ones(sz, sz)) == 1)：将上三角矩阵转换为布尔张量，
    上三角部分为 True，其余部分为 False。
    .transpose(0, 1)：将布尔张量转置，得到一个下三角矩阵，
    其中下三角部分为 True，其余部分为 False。

    .float()：将布尔张量转换为浮点数张量。
    .masked_fill(mask == 0, float('-inf'))：将布尔张量中值为 False（即 0）的位置填充为 float('-inf')。
    .masked_fill(mask == 1, float(0.0))：将布尔张量中值为 True（即 1）的位置填充为 float(0.0)。

    i.e. [[0, inf, inf],
         [0, 0, inf],
         [0, 0, 0]].
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask
        .float()
        .masked_fill(mask == 0, float('-inf'))
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
        self.fc_2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x
