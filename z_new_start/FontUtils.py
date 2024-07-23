import math
import os
import pickle
import random
import shutil
import threading

import torch
from typing import Any
from abc import ABC, abstractmethod
from einops import rearrange

from utils.judge_font import get_files
from z_new_start.FontConfig import new_start_config
from z_new_start.generate_utils.read_coordinates_pkl import draw_character_strokes


class RenderProxy:
    def __init__(self, coors):
        self._real_subject = None
        self.coors = coors
        self.path = None

    def check_access(self):
        # print("Proxy: Checking access prior to firing a real request.")
        return True

    def set_path(self, path):
        self.path = path

    def log_access(self):
        # print("Proxy: Logging the time of request.")
        pass

    def request(self):
        if self.check_access():
            if not self._real_subject:
                self._real_subject = CoorsSubject()
            self.log_access()
            ans = self._real_subject.request(self.coors, self.path)
            return ans


class Render(ABC):
    def __init__(self):
        self.degree = math.cos(1 / 3 * math.pi)

    @abstractmethod
    def renderIt(self, *arg, **kwargs: Any) -> str:
        pass

    def __call__(self, *args, **kwargs):
        try:
            gd = kwargs['gd']
        except Exception as e:
            return ['', 'coors_pkl_path']
        for i, _ in enumerate(args[0]):
            new_one = _.view(1, -1, 4).permute(1, 0, 2)
            assert not torch.isnan(new_one).any(), "NaN values in new_one"
            S, B, N = new_one.shape
            break
        x = get_files(gd.coordinate_path, gd.suffix)
        random.shuffle(x)
        try:
            coors = x[N]
        except Exception as e:
            # print(e)
            coors = x[0]
        coordinate = pickle.load(open(coors, 'rb'))
        del coordinate['font_name']
        out_path = 'Saved/samples'
        output = os.path.join(out_path, f'new.pkl')
        render = RenderProxy(coors)
        ans = client_code(render, output)
        save_images = draw_character_strokes(coordinate, scale_factor=0.25)
        for char, image in save_images.items():
            image.save(f"{out_path}/{char}.png")
        return [ans, 'coors_pkl_path']

    def __getitem__(self, *args):
        return random.random()


def _get_coors_decode(Render, **kw):
    return Render.renderIt(Render.degree, keys=kw)


def client_code(coors, output):
    coors.set_path(output)
    res = coors.request()
    return res


def get_pkl(pred, *arg, **kwargs):
    sets, x = kwargs['gd'].config_set, new_start_config
    for i, _ in enumerate(pred):
        new_one = rearrange(_, 'b s n d -> b (s n) d').permute(1, 0, 2)
        assert not torch.isnan(new_one).any(), "NaN values in new_one"
        S, B, N = new_one.shape
        break
    rl = RenderLoss()
    coors = pickle.load(open(x[sets][rl(S)[1]], 'rb'))
    del coors['font_name']
    out_path = 'Saved/samples'
    output = os.path.join(out_path, f'new.pkl')
    render = RenderProxy(coors)
    ans = client_code(render, output)
    save_images = draw_character_strokes(coors, scale_factor=0.25)
    for char, image in save_images.items():
        image.save(f"{out_path}/{char}.png")
    return ans


class RenderLoss(Render):
    def __init__(self):
        super(RenderLoss, self).__init__()

    def renderIt(self, *arg, **kwargs) -> str:
        return str(kwargs['dropout'])

    def get_mixture_coef2(self, output):
        z = output
        z_pen_logits = z[:, 0:3]

        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.split(z[:, 3:], 20, 1)

        # softmax pi weights:
        z_pi = torch.softmax(z_pi, -1)

        z_sigma1 = torch.minimum(torch.exp(z_sigma1), torch.Tensor([500.0]).cuda())
        z_sigma2 = torch.minimum(torch.exp(z_sigma2), torch.Tensor([500.0]).cuda())
        z_corr = torch.tanh(z_corr)
        result = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits]
        return result

    def get_seq_from_gmms(self, gmm_pred):
        gmm_pred = gmm_pred.reshape(-1, 123)
        [pi, mu1, mu2, sigma1, sigma2, corr, pen_logits] = self.get_mixture_coef2(gmm_pred)
        max_mixture_idx = torch.stack([torch.arange(pi.shape[0], dtype=torch.int64).cuda(), torch.argmax(pi, 1)], 1)
        next_x1 = mu1[list(max_mixture_idx.T)]
        next_x2 = mu2[list(max_mixture_idx.T)]
        pen_state = torch.argmax(gmm_pred[:, :3], dim=-1)
        pen_state = torch.nn.functional.one_hot(pen_state, num_classes=3).to(gmm_pred)
        seq_pred = torch.cat([next_x1.unsqueeze(1), next_x2.unsqueeze(1), pen_state], -1)
        return seq_pred


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


import math
import random
import pickle


class CoorsSubject:
    def __init__(self):
        self.output = {}
        self.personality = {}

    def generate_personality(self, angle_range=(-math.pi / 200, math.pi / 200),
                             scale_range=(0.998, 1.002),
                             shift_range=(-0.005, 0.005),
                             curve_range=(-0.001, 0.001)):
        self.personality = {
            'angle': random.uniform(*angle_range),
            'scale_x': random.uniform(*scale_range),
            'scale_y': random.uniform(*scale_range),
            'shift_x': random.uniform(*shift_range),
            'shift_y': random.uniform(*shift_range),
            'curve': random.uniform(*curve_range),
        }

    def apply_personality(self, x, y):
        x *= self.personality['scale_x']
        y *= self.personality['scale_y']
        nx = x * math.cos(self.personality['angle']) - y * math.sin(self.personality['angle'])
        ny = x * math.sin(self.personality['angle']) + y * math.cos(self.personality['angle'])
        nx += self.personality['shift_x']
        ny += self.personality['shift_y']
        ny += self.personality['curve'] * x * x
        return nx, ny

    def disturb_coordinate(self, x1, y1, x2, y2, degree, smooth_factor):
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return x1, y1
        dx /= length
        dy /= length

        tangent_x, tangent_y = dx, dy
        normal_x, normal_y = -dy, dx

        if smooth_factor > 1:
            mix_x = normal_x * (smooth_factor - 1)
            mix_y = normal_y * (smooth_factor - 1)
        elif 0 < smooth_factor < 1:
            mix_x = tangent_x * (1 - smooth_factor)
            mix_y = tangent_y * (1 - smooth_factor)
        else:
            mix_x, mix_y = 0, 0

        disturb_x = random.gauss(0, degree) * mix_x
        disturb_y = random.gauss(0, degree) * mix_y

        return self.apply_personality(x1 + disturb_x, y1 + disturb_y)

    def disturb_stroke(self, stroke, degree, smooth_factor):
        disturbed_stroke = []
        for i in range(len(stroke)):
            x1, y1, p1, p2 = stroke[i]
            if i == len(stroke) - 1:
                x2, y2 = stroke[0][0], stroke[0][1]
            else:
                x2, y2 = stroke[i + 1][0], stroke[i + 1][1]
            x1, y1 = self.disturb_coordinate(x1, y1, x2, y2, degree, smooth_factor)
            disturbed_stroke.append((x1, y1, p1, p2))
        return disturbed_stroke

    def request(self, x: dict, path: str, degree=0.1, smooth_factor=25,
                angle_range=(-math.pi / 100, math.pi / 100),
                scale_range=(0.998, 1.002),
                shift_range=(-20, 20),
                curve_range=(-0.00015, 0.00015)) -> str:
        self.generate_personality(angle_range, scale_range, shift_range, curve_range)

        self.output = {char: [self.disturb_stroke(stroke, degree, smooth_factor) for stroke in strokes] for
                       char, strokes in x.items()}

        with open(path, 'wb') as f:
            pickle.dump(self.output, f)
        return path


class CoorsRender(Render):
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        super(CoorsRender, self).__init__()

    def __new__(cls, *args, **kwargs):
        """
        further:
        使用锁机制实现线程安全单例模式
        :param args:
        :param kwargs:
        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(CoorsRender, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def renderIt(self, *arg, **kwargs: Any) -> str:
        if self[0] > kwargs['keys'][new_start_config['train']['keys']]:
            print("here")
            path = get_pkl(kwargs['keys']['p'], gd=kwargs['keys']['gd'])
        else:
            print('there')
            bs, _n, _, h, w = kwargs['keys']['images'].shape
            path = self(kwargs['keys']['p'], gd=kwargs['keys']['gd'], C=_)[0]
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
