import os
import shutil


def get_files(path, suffix):
    files_with_suffix = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                files_with_suffix.append(os.path.join(root, file).replace("\\", '/'))
    return files_with_suffix


save_dir_up = '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/AB_pics'
pics_path = '/mnt/data/llch/Chinese-Fonts-Dataset/lch_pics'
png_dirs = os.listdir(pics_path)
char_list = ['A', 'B', '刘']

if __name__ == '__main__':
    for png_dir in png_dirs:
        out_dir = os.path.join(save_dir_up, png_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pics_path_now = pics_path + '/' + png_dir
        a_font_pics = get_files(pics_path_now, '.png')
        for pic in a_font_pics:
            pic_name = os.path.basename(pic).split('.')[0]
            if pic_name in char_list:
                shutil.copy(pic, os.path.join(out_dir, pic_name + ".png"))

