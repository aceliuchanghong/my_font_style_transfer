import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.judge_font import get_files
import os


def resize_thin_character(pics, save_pics_path, show_pics_num, suffix='png', save_chinese_name=False,
                          erosion_level=0):
    """
    将输入图片进行处理，提取出字符骨架并显示部分结果。

    参数:
    pics (list): 图片文件路径列表
    save_pics_path (str): 处理后图片保存路径
    show_pics_num (int): 显示部分结果图片的数量
    suffix (str): 保存图片的格式，默认为 'png'
    save_chinese_name (bool): 是否使用中文名保存图片，默认为 False
    erosion_level (float): 腐蚀操作的程度，默认为 0，值越大腐蚀越强
    """
    length = len(pics)
    index = 0
    for pic in tqdm(pics, total=length):  # 使用tqdm显示处理进度
        pic_chinese_name_maybe = os.path.basename(pic).split('.')[0]
        # 读取图片为灰度图
        style_img = cv2.imdecode(np.fromfile(pic, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # 调整图片大小为64x64
        fix = cv2.resize(style_img, (64, 64))

        # 反转颜色，使前景为白色，背景为黑色
        fix = cv2.bitwise_not(fix)

        # 使用形态学操作获取骨架
        skel = np.zeros(fix.shape, np.uint8)

        # 二值化处理
        ret, fix = cv2.threshold(fix, 127, 255, cv2.THRESH_BINARY)
        # 根据fix的大小动态调整结构元素
        # element_size = max(3, int(min(fix.shape) * 0.05))  # 动态计算结构元素的大小，至少为3
        element_size = 2
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_size, element_size))

        # 使用循环进行骨架提取
        erosion_iterations = max(1, int(erosion_level * 10))
        while True:
            open = cv2.morphologyEx(fix, cv2.MORPH_OPEN, element)  # 形态学开操作
            temp = cv2.subtract(fix, open)  # 获取开操作的差异
            eroded = cv2.erode(fix, element, iterations=erosion_iterations)  # 根据腐蚀级别进行腐蚀操作
            skel = cv2.bitwise_or(skel, temp)  # 合并结果
            fix = eroded.copy()  # 更新fix

            if cv2.countNonZero(fix) == 0:  # 如果图片中所有像素都为零，跳出循环
                break

        # 反转骨架图像颜色，使骨架为黑色，背景为白色
        skel = cv2.bitwise_not(skel)

        # 如果需要展示部分结果图片
        if show_pics_num > index:
            plt.imshow(skel, cmap='gray')
            plt.show()
        # 保存处理后的图片
        if save_chinese_name:
            save_path = save_pics_path + "/" + pic_chinese_name_maybe + '.' + suffix
        else:
            save_path = os.path.join(save_pics_path, f'skel_{index}.{suffix}')
        # print(save_path)
        cv2.imencode(f'.{suffix}', skel)[1].tofile(save_path)
        index += 1


if __name__ == '__main__':
    # 设置基础目录，文件后缀名，保存路径，集数量和展示图片数量
    base_dir = '../style_samples2'
    get_suffix = ".jpg"
    save_pics_path = '../style_samples'
    show_pics_num = 2

    # 确保保存图片的目录存在
    if not os.path.exists(save_pics_path):
        os.makedirs(save_pics_path)

    # 获取文件列表
    files_list = get_files(base_dir, get_suffix)
    # 调用函数处理图片
    resize_thin_character(files_list, save_pics_path, show_pics_num, save_chinese_name=False)
