import torchvision.models as models
from models.transformer import *
from einops import rearrange
from torchvision.models.resnet import ResNet18_Weights


# 将输入的手写字体图像转换为一个可以被后续处理的特征表示
# content encoder ==> Content Transformer
class Content_TR(nn.Module):
    def __init__(self,
                 d_model=256,
                 num_head=8,
                 num_encoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True
                 ):
        super(Content_TR, self).__init__()
        self.Feat_Encoder = nn.Sequential(*(
                [nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)]
                +
                list(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[1:-2]
        ))
        encoder_layer = TransformerEncoderLayer(
            d_model, num_head, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)

    def forward(self, x):
        x = self.Feat_Encoder(x)
        """
        'n c h w -> (h w) n c'：这个字符串定义了输入张量的维度。它有四个维度：n、c、h和w。每个字母代表一个维度，它们的顺序是从最外层到最内层。
        n：通常代表批量大小（batch size），即输入数据中的样本数量。
        c：通常代表通道数（channels），即每个样本的特征数量。
        h：通常代表高度（height），即图像的高度。
        w：通常代表宽度（width），即图像的宽度。
        字符串中的箭头->表示重新排列的方向。箭头左侧是输入张量的维度，箭头右侧是输出张量的维度。

        (h w) n c：箭头右侧定义了输出张量的维度。
        这里，(h w)是一个新的维度，它是h和w的乘积，即图像的像素总数。
        n、c和h、w的顺序被交换，n现在是第二个维度，c是第三个维度，h和w的乘积是第一个维度。
        
        将一个四维张量从形状(n, c, h, w)重新排列为形状(h * w, n, c)。这样做的一个常见原因是将图像数据从二维（高度和宽度）转换为一维（像素）
        """
        x = rearrange(x, 'n c h w -> (h w) n c')
        x = self.add_position(x)
        x = self.encoder(x)
        return x


# For the training of Chinese handwriting generation task,
# we first pre-train the content encoder for character classification.
# No need to pre-train the encoder in other languages (e.g, Japanese, English and Indic).
# Content_Cls ==> Content Classifier
class Content_Cls(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_encoder_layers=3,
                 num_classes=6763
                 ) -> None:
        super(Content_Cls, self).__init__()
        self.feature_ext = Content_TR(d_model, num_encoder_layers)
        self.cls_head = nn.Linear(d_model, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.feature_ext(x)
        x = torch.mean(x, 0)
        out = self.cls_head(x)
        return out
