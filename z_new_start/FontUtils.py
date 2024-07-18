import math
import random
import torch
from typing import Any
from abc import ABC, abstractmethod

from z_new_start.FontConfig import new_start_config


class Render(ABC):
    def __init__(self):
        self.degree = random.random()

    def __call__(self, *args, **kwargs):
        print(args[0], args[1])
        return args[0]

    def __getitem__(self, *args):
        return math.cos(1 / 3 * math.pi)

    @abstractmethod
    def renderIt(self, *arg, **kwargs: Any) -> str:
        pass


def get_pkl(pred):
    for i, _ in enumerate(pred):
        # print(_.shape)
        print(_)
    return ''


def _get_coors_decode(Render, **kw):
    return Render.renderIt(Render.dice, keys=kw)


def restore_coordinates(padded_coors, max_stroke=20, max_per_stroke_point=200):
    """
    将填充后的坐标张量还原为原始的字符坐标格式。
    Args:
        padded_coors (torch.Tensor): 填充后的坐标张量，形状为 [max_stroke, max_per_stroke_point, 4]
        max_stroke (int): 最大笔画数
        max_per_stroke_point (int): 每个笔画的最大点数
    Returns:
        list: 还原后的字符坐标格式
    """
    restored_coordinates = []

    for i in range(max_stroke):
        stroke = []
        for j in range(max_per_stroke_point):
            point = padded_coors[i, j]
            if torch.all(point == 0):
                break  # 如果当前点全为0，则跳过该点
            stroke.append((point[0].item(), point[1].item(), int(point[2].item()), int(point[3].item())))
        if stroke:
            restored_coordinates.append(stroke)

    return restored_coordinates


class CoorsRender(Render):
    def __init__(self):
        super(CoorsRender, self).__init__()

    def renderIt(self, *arg, **kwargs: Any) -> str:
        if self.degree > kwargs['keys'][new_start_config['train']['keys']]:
            path = get_pkl(kwargs['keys']['pred'])
        else:
            bs, _n, _, h, w = kwargs['keys']['images'].shape
            path = self(kwargs['keys']['pred'], kwargs['keys']['gd'], _)
        return path


if __name__ == '__main__':
    # 测试示例
    padded_coors = torch.tensor([
        [[429.0, -43.0, 1.0, 0.0], [404.0, -53.0, 0.0, 0.0], [549.0, 56.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [[967.0, 744.0, 1.0, 0.0], [831.0, 774.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[704.0, 130.0, 1.0, 0.0], [760.0, 692.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ])

    restored_coordinates = restore_coordinates(padded_coors, 3, 4)
    print(restored_coordinates)
