# -*- coding: UTF-8 -*-
import os
import fontforge
import pickle


def get_character_stroke_coordinates(font_path, characters, output_dir='.'):
    """
    获取字体文件中每个字符的每个笔画的坐标点。
    font_path (str): 字体文件的路径。
    characters (str): 需要提取坐标的字符列表。
    返回：
    dict: 包含每个字符的每个笔画的坐标点，每个坐标点表示为 {char:[[(x, y, p1, p2),...],...],...}。
    x 和 y 分别表示笔画中某一个点的横坐标和纵坐标。
    p1 和 p2 是布尔标记（0 或 1），用于表示点在笔画中的角色。
    p1 表示笔画起始点，如果这个点是笔画的起始点，则 p1 的值为 1，否则为 0。
    p2 表示笔画终止点，如果这个点是笔画的终止点，则 p2 的值为 1，否则为 0。
    """
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
                    print(f"\t Stroke {stroke_index + 1}:")
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


def generate_font_from_coordinates(coordinates, output_path):
    """
    从坐标信息生成字体文件。
    coordinates (dict): 字符的坐标信息。
    output_path (str): 生成的字体文件的路径。
    """
    font = fontforge.font()
    font_name = os.path.basename(output_path)
    font.fontname = font_name

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


if __name__ == '__main__':
    """
    sudo apt-get install python3-fontforge
    /usr/bin/python3 -c "import fontforge;print(fontforge)"
    ffpython D:\\aProject\\py\\SDT\\z_new_start\\generate_utils\\gen_coordinates_pkl.py
    """
    # 字体
    test_ttf = r'D:\aProject\py\SDT\z_new_start\generate_utils\LXGWWenKaiGB-Light.ttf'
    # 获取要提取坐标的字符列表
    characters = ["刘", "一"]

    output_font_path = r'D:\aProject\py\SDT\z_new_start\generate_utils\GeneratedFont.ttf'

    coordinates = get_character_stroke_coordinates(test_ttf, characters)
    generate_font_from_coordinates(coordinates, output_font_path)
    del coordinates['font_name']
    print(coordinates)

    for k, v in coordinates.items():
        for i, char in enumerate(v):
            print(k, " stoke:", str(i + 1))
            for coor in char:
                print(coor)
