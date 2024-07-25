import math
import os
import pickle
import argparse
import numpy as np
from PIL import Image, ImageDraw
from utils.judge_font import get_files
from itertools import islice


def r_point(point, angle, center):
    x, y = point
    cx, cy = center
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x_new = cx + (x - cx) * cos_theta - (y - cy) * sin_theta
    y_new = cy + (x - cx) * sin_theta + (y - cy) * cos_theta
    return x_new, y_new


def draw_character_stroke(strokes, image_size=(256, 256), scale_factor=1, degree=0.015):
    normalized_strokes = normalize_coordinates({0: strokes}, image_size, scale_factor)[0]
    nums = int(100 * degree)
    center = (image_size[0] / 2, image_size[1] / 2)
    image = Image.new('1', image_size, 1)  # 创建黑白图像
    draw = ImageDraw.Draw(image)
    for stroke in normalized_strokes:
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
    return image


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

    return normalized


def main(opt):
    out_path = opt.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    coor_list = get_files(opt.pkl, '.pkl')
    for coors in coor_list:
        with open(coors, 'rb') as file:
            try:
                coor = pickle.load(file)
            except Exception as e:
                continue
        name = os.path.basename(coors).split('.')[0]
        coor.pop('font_name', None)

        # 获取第100个字符的坐标
        try:
            char, strokes = next(islice(coor.items(), 99, 100))
            if strokes is not None and not isinstance(strokes, str):
                image = draw_character_stroke(strokes, scale_factor=opt.scale, degree=opt.degree)
                image.save(f"{out_path}/{name}.png")
                print(f"Processed {name}")
            else:
                print(f"Invalid strokes data for {name}")
        except StopIteration:
            print(f"Less than 100 characters in {name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='files/coors_pics_path', help='输出图片目录')
    parser.add_argument('--pkl', default=r'D:\download\font_coor\lch_coor', help='读取文件')
    parser.add_argument('--scale', default=0.2, type=float, help='图片缩放尺寸')
    parser.add_argument('--degree', default=0.015, type=float)
    opt = parser.parse_args()
    main(opt)
