# -*- coding: UTF-8 -*-
import os
import fontforge
import concurrent.futures
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_character_stroke_coordinates2(font_path, characters, output_dir='.'):
    coordinates = {}
    # 打开字体文件
    font = fontforge.open(font_path)
    # 遍历每个字符并生成坐标点
    for char in characters:
        try:
            glyph = font[ord(char)]
            print(f"Processing: {char}")
            if glyph.isWorthOutputting():  # 检查字符是否存在
                char_coords = []
                for stroke_index, contour in enumerate(glyph.foreground):
                    # print(f"\t Stroke {stroke_index + 1}:")
                    stroke_coords = []
                    num_points = len(contour)
                    for i, point in enumerate(contour):
                        x, y = point.x, point.y
                        p1 = 1 if i == 0 else 0
                        p2 = 1 if i == num_points - 1 else 0
                        stroke_coords.append((x, y, p1, p2))
                    char_coords.append(stroke_coords)
                coordinates[char] = char_coords
            else:
                print("字符不存在:", char)
                coordinates[char] = None  # 字符不存在
        except Exception as e:
            coordinates[char] = f"Error: {e}"

    # 关闭字体文件
    font.close()
    # 字体坐标讯息存储为pkl
    font_name = os.path.basename(font_path).split('.')[0]
    coordinates["font_name"] = font_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pickle.dump(coordinates, open(output_dir + "/" + font_name + '.pkl', 'wb'))

    return coordinates


def get_ttf_files(directory):
    """
    递归地遍历指定目录及其子目录，找到所有 .ttf 文件，并返回这些文件的绝对路径列表。

    :param directory: 要遍历的目录路径
    :return: 所有 .ttf 文件的绝对路径列表
    """
    ttf_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ttf'):
                ttf_files.append(os.path.abspath(os.path.join(root, file)))

    return ttf_files


if __name__ == '__main__':

    # nohup /usr/bin/python3 4_gen_coors_pkl.py > 4coor.log &

    pkl_path = '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/new_character_dict.pkl'
    output_dir = 'AB_coors'
    ttf_dir = '/mnt/data/llch/Chinese-Fonts-Dataset/z_ttf'

    char_dict = pickle.load(open(pkl_path, 'rb'))
    ttf_list = get_ttf_files(ttf_dir)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_character_stroke_coordinates2, font_path, char_dict, output_dir) for font_path in
                   ttf_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
