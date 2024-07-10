import os

from utils.judge_font import get_files
from utils.util import write_pkl

char_pics_path = r'D:\soft\FontForgeBuilds\LCH_pics\00'
# char_pics_path = '/mnt/data/llch/Chinese-Fonts-Dataset/lch_pics'
get_suffix = '.png'
save_pkl_file_path = 'lch_pics_pkl'
if not os.path.exists(save_pkl_file_path):
    os.makedirs(save_pkl_file_path)

if __name__ == '__main__':
    """
    将所有字体转png后==>存储文字的图像
    """
    # 获取文件列表
    png_dirs = os.listdir(char_pics_path)
    for png_dir in png_dirs:
        imgs_file_list = get_files(os.path.join(char_pics_path, png_dir), get_suffix)
        save_pkl_file_name = png_dir + '.pkl'
        write_pkl(save_pkl_file_path, save_pkl_file_name, imgs_file_list, 0)
