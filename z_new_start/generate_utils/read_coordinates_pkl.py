import math
import os
import pickle
import argparse
import numpy as np
from PIL import Image, ImageDraw


def r_point(point, angle, center):
    x, y = point
    cx, cy = center
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x_new = cx + (x - cx) * cos_theta - (y - cy) * sin_theta
    y_new = cy + (x - cx) * sin_theta + (y - cy) * cos_theta
    return x_new, y_new


def draw_character_strokes(coordinates, image_size=(256, 256), scale_factor=1, degree=0.0):
    normalized_coordinates = normalize_coordinates(coordinates, image_size, scale_factor)
    images = {}
    for char, strokes in normalized_coordinates.items():
        if strokes is None or isinstance(strokes, str):
            continue
        nums = int(100 * degree)
        center = (image_size[0] / 2, image_size[1] / 2)
        image = Image.new('1', image_size, 1)  # 创建黑白图像
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x1, y1, _, _ = stroke[i]
                x2, y2, _, _ = stroke[i + 1]
                temp1 = np.random.rand()
                temp2 = np.random.rand()
                if temp1 < degree:
                    continue
                if temp2 < degree:
                    x1 += np.random.randint(-nums, nums)
                    y1 += np.random.randint(-nums, nums)
                    x2 += np.random.randint(-nums, nums)
                    y2 += np.random.randint(-nums, nums)
                r_angle = math.pi * degree * (2 * np.random.rand() - 1)
                x1, y1 = r_point((x1, y1), r_angle, center)
                x2, y2 = r_point((x2, y2), r_angle, center)
                draw.line(
                    (x1, y1, x2, y2),
                    fill=0, width=2  # 使用黑色线条
                )
                # 添加调试信息
                # print(f"Drawing line: ({x1}, {y1}) -> ({x2}, {y2})")
        images[char] = image
    return images


def normalize_coordinates(coordinates, image_size, scale_factor):
    normalized = {}
    for char, strokes in coordinates.items():
        if strokes is None or isinstance(strokes, str):
            continue

        all_points = [point for stroke in strokes for point in stroke]
        min_x = min(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_x = max(point[0] for point in all_points)
        max_y = max(point[1] for point in all_points)

        width = max_x - min_x
        height = max_y - min_y

        offset_x = (image_size[0] - width * scale_factor) / 2
        offset_y = (image_size[1] - height * scale_factor) / 2

        norm_strokes = []
        for stroke in strokes:
            norm_stroke = [
                ((x - min_x) * scale_factor + offset_x, (image_size[1] - (y - min_y) * scale_factor - offset_y), p1, p2)
                for x, y, p1, p2 in stroke]
            norm_strokes.append(norm_stroke)
        normalized[char] = norm_strokes

        # 添加调试信息
        # print(f"Character: {char}")
        # print(f"Original strokes: {strokes}")
        # print(f"Normalized strokes: {norm_strokes}")

    return normalized


def main(opt):
    out_path = opt.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    coor = pickle.load(open(opt.pkl, 'rb'))
    del coor['font_name']
    images = draw_character_strokes(coor, scale_factor=opt.scale, degree=opt.degree)
    for char, image in images.items():
        image.save(f"{out_path}/{char}.png")  # 保存图像


if __name__ == '__main__':
    """
    conda activate SDTLog1
    cd z_new_start/generate_utils
    python read_coordinates_pkl.py
    python read_coordinates_pkl.py --pkl ../ABtest/files/AB_coors/AliHYAiHei.pkl
    python read_coordinates_pkl.py --pkl ../ABtest/files/LXGWWenKaiGB-Light.pkl
    python read_coordinates_pkl.py --pkl ../ABtest/files/LXGWWenKaiGB-Light.pkl --degree 0.03
    python read_coordinates_pkl.py --pkl ../ABtest/files/AB_coors/HYXiDengXianJ.pkl --degree 0.03
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='files/coors_pics_path', help='输出图片目录')
    parser.add_argument('--pkl', default='new_std_coor.pkl', help='读取文件')
    parser.add_argument('--scale', default=0.27, type=float, help='图片缩放尺寸')
    parser.add_argument('--degree', default=0.05, type=float)
    opt = parser.parse_args()
    main(opt)
