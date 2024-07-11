import torch.nn as nn


class FontLoss(nn.Module):
    def __init__(self, coordinate_weight=1.0, stroke_weight=0.5):
        super(FontLoss, self).__init__()
        self.coordinate_weight = coordinate_weight
        self.stroke_weight = stroke_weight
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, pred, target):
        # 假设 pred 和 target 的形状是 [batch_size, seq_len, 4]
        # 其中最后一维的4个值分别表示 [x, y, stroke_start, stroke_end]

        # 坐标损失 (x, y)
        coordinate_loss = self.mse_loss(pred[..., :2], target[..., :2])
        # 笔画信息损失 (stroke_end, stroke_start)
        stroke_loss = self.l1_loss(pred[..., 2:], target[..., 2:])
        # 总损失
        total_loss = self.coordinate_weight * coordinate_loss + self.stroke_weight * stroke_loss

        return total_loss


if __name__ == '__main__':
    criterion = FontLoss(coordinate_weight=1.0, stroke_weight=0.5)
