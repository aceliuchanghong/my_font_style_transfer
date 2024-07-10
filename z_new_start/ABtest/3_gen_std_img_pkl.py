from utils.judge_font import get_files
from utils.util import write_pkl

std_char_path = r'D:\aProject\py\SDT\z_new_start\ABtest\files\AB_pics'
get_suffix = '.png'

if __name__ == '__main__':
    """
    将字体转png后==>存储标准文字的图像
    """
    # 获取文件列表
    imgs_file_list = get_files(std_char_path, get_suffix)
    # 存储参数
    save_pkl_file_path = '.'
    save_pkl_file_name = 'files/new_chinese_content.pkl'
    pics_path = std_char_path
    write_pkl(save_pkl_file_path, save_pkl_file_name, imgs_file_list, 2)
