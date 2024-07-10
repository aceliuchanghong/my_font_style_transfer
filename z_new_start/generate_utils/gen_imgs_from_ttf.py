import pickle
import os
import fontforge
import json
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
output_dir = 'font_pics'
test_ttf = r'D:\aProject\py\SDT\z_new_start\generate_utils\LXGWWenKaiGB-Light.ttf'
char_json_path = r'D:\aProject\py\SDT\tf_test\txt9169.json'
new_char_dic = r'D:\aProject\py\SDT\z_new_start\generate_utils\new_character_dict.pkl'
char_dict = pickle.load(open(new_char_dic, 'rb'))
char_dict = ['A', 'B', '一']


def draw_glyph(font_path, char_dict):
    try:
        try:
            font = fontforge.open(font_path)
        except Exception as e:
            print(f"Error open {font_path}: {e}")
        font.em = 256
        # 输出图片路径=字体名称+生成数量
        output_subdir = os.path.join(output_dir, os.path.basename(font_path).split('.')[0] + str(len(char_dict)))
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        with open(char_json_path, 'r', encoding='utf-8') as f:
            cjk = json.load(f)
        cn_charset = cjk["gbk"]

        for char in char_dict:
            if char in cn_charset:
                logger.info(f"generating: {char}")
                try:
                    glyph = font[ord(char)]  # Get the glyph for the character
                    glyph.export(os.path.join(output_subdir, f"{char}.png"), 255)
                except Exception as e:
                    logger.error(f"Glyph not found for character {char}: {e}")
            else:
                logger.info("no conclude:", char[0])
    except Exception as e:
        logger.error(f"Error processing font {font_path}: {e}")


if __name__ == '__main__':
    """
    ffpython D:\\aProject\\py\\SDT\\z_new_start\\generate_utils\\gen_imgs_from_ttf.py
    """
    try:
        draw_glyph(test_ttf, char_dict)
    except Exception as e:
        print(f"Error01: {e}")
