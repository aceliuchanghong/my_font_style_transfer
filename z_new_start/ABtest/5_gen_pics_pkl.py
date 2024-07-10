import os
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_files(path, suffix):
    files_with_suffix = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                files_with_suffix.append(os.path.join(root, file).replace("\\", '/'))
    return files_with_suffix


def write_pkl(file_path, file_name, imgs_path, show_pic_num=0):
    """
    file_path pkl位置
    file_name pkl名字
    imgs_path 图片list
    show_pic_num 展示图片数量
    """
    img_list = []
    index = 0
    for img_path in imgs_path:
        img_dic = {}
        label = os.path.basename(img_path).split('.')[:-1][0]
        style_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        style_img = cv2.resize(style_img, (64, 64))
        img_dic['img'] = style_img
        img_dic['label'] = label

        if show_pic_num > index:
            plt.imshow(img_dic['img'], cmap='gray')
            plt.show()
        index += 1
        img_list.append(img_dic)
    pickle.dump(img_list, open(os.path.join(file_path, file_name), 'wb'))
    return img_list


char_pics_path = '/mnt/data/llch/Chinese-Fonts-Dataset/all_test/AB_pics'
get_suffix = '.png'
save_pkl_file_path = 'AB_pics_pkl'
if not os.path.exists(save_pkl_file_path):
    os.makedirs(save_pkl_file_path)

if __name__ == '__main__':
    """
    将所有字体转png后==>存储文字的图像
    python 5_gen_pics_pkl.py
    """
    # 获取文件列表
    png_dirs = os.listdir(char_pics_path)
    for png_dir in png_dirs:
        imgs_file_list = get_files(os.path.join(char_pics_path, png_dir), get_suffix)
        save_pkl_file_name = png_dir + '.pkl'
        write_pkl(save_pkl_file_path, save_pkl_file_name, imgs_file_list, 0)
