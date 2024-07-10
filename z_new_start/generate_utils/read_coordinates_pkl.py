import os
import pickle
import argparse

from PIL import Image, ImageDraw


def draw_character_strokes(coordinates, image_size=(256, 256), scale_factor=1):
    normalized_coordinates = normalize_coordinates(coordinates, image_size, scale_factor)
    images = {}
    for char, strokes in normalized_coordinates.items():
        if strokes is None or isinstance(strokes, str):
            continue

        image = Image.new('1', image_size, 1)  # 创建黑白图像
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x1, y1, _, _ = stroke[i]
                x2, y2, _, _ = stroke[i + 1]
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
        print(f"Character: {char}")
        print(f"Original strokes: {strokes}")
        print(f"Normalized strokes: {norm_strokes}")

    return normalized


def main(opt):
    out_path = opt.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    coor = pickle.load(open(opt.pkl, 'rb'))
    del coor['font_name']
    images = draw_character_strokes(coor, scale_factor=opt.scale)
    for char, image in images.items():
        image.save(f"{out_path}/{char}.png")  # 保存图像


if __name__ == '__main__':
    """
    conda activate SDTLog1
    cd z_new_start/generate_utils
    python read_coordinates_pkl.py
    python read_coordinates_pkl.py --pkl ../ABtest/files/AB_coors/AliHYAiHei.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='coors_test_path', help='输出图片目录')
    parser.add_argument('--pkl', default='../ABtest/files/AB_coors/HYAlzheimer.pkl', help='读取文件')
    parser.add_argument('--scale', default=0.27, type=int, help='图片缩放尺寸')
    opt = parser.parse_args()
    main(opt)
