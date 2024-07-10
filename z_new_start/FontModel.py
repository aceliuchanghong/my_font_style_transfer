import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from models.model import SeqtoEmb, EmbtoSeq
from models.transformer import *
from models.encoder import Content_TR


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
                 ):
        super(FontModel, self).__init__()

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
        self.SeqtoEmb = SeqtoEmb(output_dim=d_model)
        self.EmbtoSeq = EmbtoSeq(input_dim=d_model)
        self.add_position = PositionalEncoding(dim=d_model, dropout=0.1)
        # 参数重置 用于初始化模型的参数
        self._reset_parameters()

    def _reset_parameters(self):
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

    def forward(self, char_img_gt, std_coors):
        # 特征提取
        feat = self.feat_encoder(char_img_gt)  # 提取图像特征
        feat = feat.flatten(2).permute(2, 0, 1)  # 重塑特征以适应Transformer的输入格式
        # 编码图像特征
        encoded_feat = self.base_encoder(feat)  # 基本编码器
        # 字形编码
        glyph_feat = self.glyph_encoder(encoded_feat)  # 字形编码器

        # 内容编码
        content_feat = self.content_encoder(std_coors)  # 使用标准坐标进行内容编码

        # 字形解码
        glyph_decoded = self.glyph_transformer_decoder(content_feat, glyph_feat)  # 字形Transformer解码器

        # 通过MLP生成最终的字符输出
        character_output = self.pro_mlp_character(glyph_decoded)  # 最终的字符输出

        # 返回最终的字符输出
        return character_output

    def inference(self, img_list):
        self.eval()  # 切换到评估模式
        outputs = []

        with torch.no_grad():  # 禁用梯度计算以提高推理速度并节省内存
            for img in img_list:
                img = img.unsqueeze(0)  # 增加批次维度

                pass

        return outputs
