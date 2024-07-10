import torchvision.models as models
from models.transformer import *
from models.encoder import Content_TR
from einops import rearrange, repeat
from models.gmm import get_seq_from_gmm
from torchvision.models.resnet import ResNet18_Weights

'''
the overall architecture of our style-disentangled Transformer (SDT).
the input of our SDT is the gray image with 1 channel.
'''

"""
d_model: 模型的输入和输出的维度。在Transformer模型中，这通常是词嵌入的维度。

nhead: 多头注意力机制中的头数。在Transformer模型中，多头注意力允许模型在不同的表示子空间中并行处理信息。

num_encoder_layers: 编码器层的数量，即Transformer模型中的Transformer层数。

num_head_layers: 在编码器和解码器中使用的多头注意力层的数量。

wri_dec_layers: 书写解码器的层。同一作者对于不同字的风格

gly_dec_layers: 笔画解码器的层。同一个字不同作者的风格

dim_feedforward: 前馈神经网络中隐藏层的大小。

dropout: 在模型中使用的dropout概率，用于正则化和防止过拟合。

activation: 前馈神经网络中使用的激活函数的类型。

normalize_before: 一个布尔值，指示在应用多头注意力和前馈神经网络之前是否对输入进行层归一化。

return_intermediate_dec: 一个布尔值，指示是否在解码过程中返回中间结果。
"""


class SDT_Generator(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=2,
                 num_head_layers=1,
                 wri_dec_layers=2,
                 gly_dec_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,
                 return_intermediate_dec=True):
        super(SDT_Generator, self).__init__()
        """
        # 假设我们有一些模块列表
        modules = [
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # ... 其他模块
        ]
        
        # 直接将列表传递给 nn.Sequential
        model = nn.Sequential(modules)
        """
        # style encoder with dual heads
        # Feat_Encoder 是一个卷积层和一个预训练的 ResNet-18 模型的特征提取器，它可以用于提取图像的特征
        self.Feat_Encoder = nn.Sequential(*(  # 一个输入通道，输出64个通道。卷积核大小为7，步长为2，填充为3。bias 设置为 False 表示不使用偏置项。
                [nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)]
                +
                # 获取了 ResNet-18 模型的子模块列表，然后去掉了列表的第一个和最后两个模块。
                # 这些被去掉的模块通常是 ResNet-18 模型的头部，包括全局平均池化层和全连接层。
                # 从列表的第二个元素开始，一直到倒数第三个元素结束，不包括这两个元素
                list(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[1:-2]
        ))
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        # Transformer 编码器 用于对输入的特征进行编码，以提取全局的风格信息。
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        writer_norm = nn.LayerNorm(d_model) if normalize_before else None
        glyph_norm = nn.LayerNorm(d_model) if normalize_before else None
        # Writer Head 和 Glyph Head: 这两个头部分别用于提取作者风格和字符风格。
        # 它们使用相同的 Transformer 编码器层，但可以有不同的层数。
        self.writer_head = TransformerEncoder(encoder_layer, num_head_layers, writer_norm)
        self.glyph_head = TransformerEncoder(encoder_layer, num_head_layers, glyph_norm)

        # content encoder
        # 内容编码器 用于对输入的内容进行编码，以提取内容信息。
        self.content_encoder = Content_TR(d_model=d_model, num_encoder_layers=num_encoder_layers)

        # decoder for receiving writer-wise and character-wise styles
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        wri_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.wri_decoder = TransformerDecoder(
            decoder_layer, wri_dec_layers, wri_decoder_norm, return_intermediate=return_intermediate_dec
        )
        gly_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.gly_decoder = TransformerDecoder(
            decoder_layer, gly_dec_layers, gly_decoder_norm, return_intermediate=return_intermediate_dec
        )

        # two 多层感知器(mlp)s that project style features into the space where nce_loss is applied
        # 用于将提取的风格特征投影到一个较低维度的空间
        # 将风格特征从高维空间投影到一个较低维度的空间。这个过程通常被称为特征提取或降维，因为它可以帮助模型更有效地处理数据。
        self.pro_mlp_writer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.GELU(),
            nn.Linear(4096, 256)
        )
        self.pro_mlp_character = nn.Sequential(
            nn.Linear(512, 4096),
            nn.GELU(),
            nn.Linear(4096, 256)
        )
        # 序列到嵌入 (SeqtoEmb) 和 嵌入到序列 (EmbtoSeq)
        # 这两个模块用于处理序列数据和嵌入数据之间的转换。
        self.SeqtoEmb = SeqtoEmb(output_dim=d_model)
        self.EmbtoSeq = EmbtoSeq(input_dim=d_model)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
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

    def random_double_sampling(self, x, ratio=0.25):
        """
        它用于在序列数据中进行双重采样。双重采样通常用于训练序列模型
        通过随机排序每个组内的序列元素来生成锚点和正样本，从而在训练序列模型时提供正样本对。

        例如Transformer，以便在每个批次中生成正样本对（例如，一个输入序列和其对应的目标序列）。
        random_double_sampling方法接受一个四维的输入张量x，形状为[L, B, N, D]，其中：
        L是序列的长度。
        B是批次大小。
        N是每个序列中要采样的组数。
        D是每个组的维度。

        Sample the positive pair (i.e., o and o^+) within a character by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [L, B, N, D], sequence
        return o [B, N, 1, D], o^+ [B, N, 1, D]
        """
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

    # the shape of style_imgs is [B, 2*N, C, H, W] during training
    def forward(self, style_imgs, seq, char_img):
        # style_imgs 是风格图片的输入，seq 是序列输入，char_img 是字符图片输入。
        # 风格图片的批次大小、图片数量、通道数、高度和宽度。
        batch_size, num_imgs, in_planes, h, w = style_imgs.shape

        # style_imgs: [B, 2*N, C:1, H, W] -> FEAT_ST_ENC: [4*N, B, C:512]
        # -1是一个特殊的值，表示该维度的大小将通过其他维度的大小和总元素数自动推断出来
        # 调整风格图像的形状
        style_imgs = style_imgs.view(-1, in_planes, h, w)  # [B*2N, C:1, H, W]
        style_embe = self.Feat_Encoder(style_imgs)  # [B*2N, C:512, 2, 2]

        anchor_num = num_imgs // 2
        # [4, B*2N, C:512] permute,改变张量的维度顺序
        style_embe = style_embe.view(batch_size * num_imgs, 512, -1).permute(2, 0, 1)
        FEAT_ST_ENC = self.add_position(style_embe)

        memory = self.base_encoder(FEAT_ST_ENC)  # [4, B*2N, C]
        writer_memory = self.writer_head(memory)
        glyph_memory = self.glyph_head(memory)

        writer_memory = rearrange(writer_memory, 't (b p n) c -> t (p b) n c',
                                  b=batch_size, p=2, n=anchor_num)  # [4, 2*B, N, C]
        glyph_memory = rearrange(glyph_memory, 't (b p n) c -> t (p b) n c',
                                 b=batch_size, p=2, n=anchor_num)  # [4, 2*B, N, C]

        # writer-nce
        memory_fea = rearrange(writer_memory, 't b n c ->(t n) b c')  # [4*N, 2*B, C]
        # 计算memory_fea张量在第0个维度上的平均值
        compact_fea = torch.mean(memory_fea, 0)  # [2*B, C]
        # compact_fea:[2*B, C:512] ->  nce_emb: [B, 2, C:128]
        pro_emb = self.pro_mlp_writer(compact_fea)
        query_emb = pro_emb[:batch_size, :]
        pos_emb = pro_emb[batch_size:, :]
        # 将两个嵌入向量（query_emb和pos_emb）沿着第二个维度（索引为1）堆叠起来，形成一个新的张量
        nce_emb = torch.stack((query_emb, pos_emb), 1)  # [B, 2, C]
        nce_emb = nn.functional.normalize(nce_emb, p=2, dim=2)

        # glyph-nce
        patch_emb = glyph_memory[:, :batch_size]  # [4, B, N, C]
        # sample the positive pair
        anc, positive = self.random_double_sampling(patch_emb)
        n_channels = anc.shape[-1]
        # -1：这是一个特殊的值，表示该维度的大小由其他维度和总元素数量决定
        anc = anc.reshape(batch_size, -1, n_channels)
        # 如果anc是一个形状为(m, n)的二维张量，
        # 那么torch.mean(anc, 1, keepdim=True)将返回一个形状为(m, 1)的二维张量，
        # 其中每个元素是原始张量对应行的均值
        anc_compact = torch.mean(anc, 1, keepdim=True)
        anc_compact = self.pro_mlp_character(anc_compact)  # [B, 1, C]
        positive = positive.reshape(batch_size, -1, n_channels)
        positive_compact = torch.mean(positive, 1, keepdim=True)
        positive_compact = self.pro_mlp_character(positive_compact)  # [B, 1, C]

        nce_emb_patch = torch.cat((anc_compact, positive_compact), 1)  # [B, 2, C]
        nce_emb_patch = nn.functional.normalize(nce_emb_patch, p=2, dim=2)

        # input the writer-wise & character-wise styles into the decoder
        writer_style = memory_fea[:, :batch_size, :]  # [4*N, B, C]
        glyph_style = glyph_memory[:, :batch_size]  # [4, B, N, C]
        glyph_style = rearrange(glyph_style, 't b n c -> (t n) b c')  # [4*N, B, C]

        # QUERY: [char_emb, seq_emb]
        seq_emb = self.SeqtoEmb(seq).permute(1, 0, 2)
        T, N, C = seq_emb.shape

        char_emb = self.content_encoder(char_img)  # [4, N, 512]
        char_emb = torch.mean(char_emb, 0)  # [N, 512]
        char_emb = repeat(char_emb, 'n c -> t n c', t=1)
        tgt = torch.cat((char_emb, seq_emb), 0)  # [1+T], put the content token as the first token
        tgt_mask = generate_square_subsequent_mask(sz=(T + 1)).to(tgt)
        tgt = self.add_position(tgt)

        # [wri_dec_layers, T, B, C]
        wri_hs = self.wri_decoder(tgt, writer_style, tgt_mask=tgt_mask)
        # [gly_dec_layers, T, B, C]
        hs = self.gly_decoder(wri_hs[-1], glyph_style, tgt_mask=tgt_mask)

        # 将矩阵hs的第二和第三维度进行转置
        h = hs.transpose(1, 2)[-1]  # B T C
        pred_sequence = self.EmbtoSeq(h)
        return pred_sequence, nce_emb, nce_emb_patch

    # style_imgs: [B, N, C, H, W]
    def inference(self, style_imgs, char_img, max_len):
        batch_size, num_imgs, in_planes, h, w = style_imgs.shape
        # [B, N, C, H, W] -> [B*N, C, H, W]
        style_imgs = style_imgs.view(-1, in_planes, h, w)
        # [B*N, 1, 64, 64] -> [B*N, 512, 2, 2]
        style_embe = self.Feat_Encoder(style_imgs)
        FEAT_ST = style_embe.reshape(batch_size * num_imgs, 512, -1).permute(2, 0, 1)  # [4, B*N, C]
        FEAT_ST_ENC = self.add_position(FEAT_ST)  # [4, B*N, C:512]
        memory = self.base_encoder(FEAT_ST_ENC)  # [5, B*N, C]
        memory_writer = self.writer_head(memory)  # [4, B*N, C]
        memory_glyph = self.glyph_head(memory)  # [4, B*N, C]
        memory_writer = rearrange(
            memory_writer, 't (b n) c ->(t n) b c', b=batch_size
        )  # [4*N, B, C]
        memory_glyph = rearrange(
            memory_glyph, 't (b n) c -> (t n) b c', b=batch_size
        )  # [4*N, B, C]

        char_emb = self.content_encoder(char_img)
        char_emb = torch.mean(char_emb, 0)  # [N, 256]
        src_tensor = torch.zeros(max_len + 1, batch_size, 512).to(char_emb)
        pred_sequence = torch.zeros(max_len, batch_size, 5).to(char_emb)
        src_tensor[0] = char_emb
        tgt_mask = generate_square_subsequent_mask(sz=max_len + 1).to(char_emb)
        for i in range(max_len):
            src_tensor[i] = self.add_position(src_tensor[i], step=i)

            wri_hs = self.wri_decoder(
                src_tensor, memory_writer, tgt_mask=tgt_mask
            )
            hs = self.gly_decoder(
                wri_hs[-1], memory_glyph, tgt_mask=tgt_mask
            )

            output_hid = hs[-1][i]
            gmm_pred = self.EmbtoSeq(output_hid)
            pred_sequence[i] = get_seq_from_gmm(gmm_pred)
            pen_state = pred_sequence[i, :, 2:]
            seq_emb = self.SeqtoEmb(pred_sequence[i])
            src_tensor[i + 1] = seq_emb
            if sum(pen_state[:, -1]) == batch_size:
                break
            else:
                pass
        return pred_sequence.transpose(0, 1)  # N, T, C        


class SeqtoEmb(nn.Module):
    """
    project the handwriting sequences to the transformer hidden space
    """

    def __init__(self, output_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(5, 256)
        self.fc_2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x


class EmbtoSeq(nn.Module):
    """
    project the transformer hidden space to handwriting sequences
    """

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 256)
        self.fc_2 = nn.Linear(256, 123)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    generate the attention mask, i.e. [[0, inf, inf],
                                       [0, 0, inf],
                                       [0, 0, 0]].
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask
        .float()
        .masked_fill(mask == 0, float('-inf'))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask
