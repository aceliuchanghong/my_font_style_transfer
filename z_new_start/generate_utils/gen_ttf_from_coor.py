# -*- coding: UTF-8 -*-
import os
import fontforge
import pickle
import argparse


def gen_ttf_from_coors(coordinates, output_path, font_name):
    """
    从坐标信息生成字体文件。
    coordinates (dict): 字符的坐标信息。
    output_path (str): 生成的字体文件的路径。
    font_name (str): 字体的名称。
    """
    font = fontforge.font()
    font.fontname = font_name
    font.fullname = font_name
    font.familyname = font_name

    for char, char_coords in coordinates.items():
        if char == "font_name" or not char_coords:
            continue
        glyph = font.createChar(ord(char))
        pen = glyph.glyphPen()
        for stroke in char_coords:
            pen.moveTo(stroke[0][:2])
            for point in stroke[1:]:
                pen.lineTo(point[:2])
            pen.closePath()
        glyph.width = 1000  # 设置字符宽度，可以根据需要调整

    font.generate(output_path)
    print(f"Generated font at {output_path} with name {font_name}")


def main(opt):
    output_path = opt.out
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    coors = pickle.load(open(opt.pkl, 'rb'))
    output = os.path.join(output_path, opt.name)
    gen_ttf_from_coors(coors, output, opt.name)


if __name__ == '__main__':
    """
    ffpython D:\\aProject\\py\\my_font_style_transfer\\z_new_start\\generate_utils\\gen_ttf_from_coor.py
    ffpython D:\aProject\py\my_font_style_transfer\z_new_start\generate_utils\gen_ttf_from_coor.py --pkl D:\aProject\py\my_font_style_transfer\z_new_start\generate_utils\files\coors_pics_path\processed_coordinates.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=r'D:\aProject\py\my_font_style_transfer\Saved\samples',
                        help='输出目录')
    parser.add_argument('--pkl', default=r'D:\aProject\py\my_font_style_transfer\Saved\samples\new.pkl',
                        help='来源文件')
    parser.add_argument('--name', default='new.ttf', help='文件名字')
    opt = parser.parse_args()
    main(opt)
