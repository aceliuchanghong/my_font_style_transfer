import numpy as np
import torch
import random
from PIL import ImageDraw, Image
import cv2
import os
import pickle
import matplotlib.pyplot as plt

'''
description: Normalize the xy-coordinates into a standard interval.
Refer to "Drawing and Recognizing Chinese Characters with Recurrent Neural Network".
'''


def normalize_xys(xys):
    stroken_state = np.cumsum(np.concatenate((np.array([0]), xys[:, -2]))[:-1])
    px_sum = py_sum = len_sum = 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_px, temp_py = temp_len * (xy_1 + xy) / 2
            px_sum += temp_px
            py_sum += temp_py
            len_sum += temp_len
    if len_sum == 0:
        raise Exception("Broken online characters")
    else:
        pass

    mux, muy = px_sum / len_sum, py_sum / len_sum
    dx_sum, dy_sum = 0, 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_dx = temp_len * (
                    np.power(xy_1[0] - mux, 2) + np.power(xy[0] - mux, 2) + (xy_1[0] - mux) * (xy[0] - mux)) / 3
            temp_dy = temp_len * (
                    np.power(xy_1[1] - muy, 2) + np.power(xy[1] - muy, 2) + (xy_1[1] - muy) * (xy[1] - muy)) / 3
            dx_sum += temp_dx
            dy_sum += temp_dy
    sigma = np.sqrt(dx_sum / len_sum)
    if sigma == 0:
        sigma = np.sqrt(dy_sum / len_sum)
    xys[:, 0], xys[:, 1] = (xys[:, 0] - mux) / sigma, (xys[:, 1] - muy) / sigma
    return xys


'''
description: Rendering offline character images by connecting coordinate points
'''


def coords_render(coordinates, split, width, height, thickness, board=5):
    canvas_w = width
    canvas_h = height
    board_w = board
    board_h = board
    # preprocess canvas size
    p_canvas_w = canvas_w - 2 * board_w
    p_canvas_h = canvas_h - 2 * board_h

    # find original character size to fit with canvas
    min_x = 635535
    min_y = 635535
    max_x = -1
    max_y = -1

    coordinates[:, 0] = np.cumsum(coordinates[:, 0])
    coordinates[:, 1] = np.cumsum(coordinates[:, 1])
    if split:
        ids = np.where(coordinates[:, -1] == 1)[0]
        if len(ids) < 1:  # if not exist [0, 0, 1]
            ids = np.where(coordinates[:, 3] == 1)[0] + 1
            if len(ids) < 1:  # if not exist [0, 1, 0]
                ids = np.array([len(coordinates)])
                xys_split = np.split(coordinates, ids, axis=0)[:-1]  # remove the blank list
            else:
                xys_split = np.split(coordinates, ids, axis=0)
        else:  # if exist [0, 0, 1]
            remove_end = np.split(coordinates, ids, axis=0)[0]
            ids = np.where(remove_end[:, 3] == 1)[0] + 1  # break in [0, 1, 0]
            xys_split = np.split(remove_end, ids, axis=0)
    else:
        xys_split = None
        pass
    for stroke in xys_split:
        for (x, y) in stroke[:, :2].reshape((-1, 2)):
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
    original_size = max(max_x - min_x, max_y - min_y)
    canvas = Image.new(mode='L', size=(canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        xys = np.stack([xs, ys], axis=-1).reshape(-1)
        xys[::2] = (xys[::2] - min_x) / original_size * p_canvas_w + board_w
        xys[1::2] = (xys[1::2] - min_y) / original_size * p_canvas_h + board_h
        xys = np.round(xys)
        draw.line(xys.tolist(), fill=0, width=thickness)
    # canvas 返回的是一个 PIL（Pillow）图像对象。这个图像对象是一个灰度图像，大小为 (width, height)，
    # 其中包含了根据输入坐标点绘制的线条。图像的背景色为白色（255），线条颜色为黑色（0），线条的宽度由 thickness 参数决定。
    return canvas


# fix random seeds for reproducibility
def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)


# model loads specific parameters (i.e., par) from pretrained_model
def load_specific_dict(model, pretrained_model, par):
    # 它获取模型的当前状态字典（model.state_dict()），这包含了模型的所有参数
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    # 检查 `par` 是否在预训练模型参数字典的第一个键中
    if par in list(pretrained_dict.keys())[0]:
        # 计算 `par` 字符串的长度并加1，用于截取键
        count = len(par) + 1
        # 更新预训练模型参数字典，只保留键在 `count` 之后部分在当前模型参数字典中的键值对
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        # 如果 `par` 不在第一个键中，则直接过滤保留键在当前模型参数字典中的键值对
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 如果预训练参数字典不为空，则更新当前模型的参数字典
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


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


def dxdynp_to_list(coordinates):
    """
    description: convert the np version of coordinates to the list counterpart
    将一个包含坐标信息的 NumPy 数组分割成多个笔画，并将每个笔画的坐标序列存储在一个列表中，同时计算并返回一个长度值
    coord_list = [array([x1, y1, x2, y2, ..., xn, yn]), array([x1, y1, x2, y2, ..., xm, ym]), ...]
    length = float_value
    """
    # 查找 coordinates 数组中最后一列等于1的索引，结果存储在 ids 中
    ids = np.where(coordinates[:, -1] == 1)[0]
    # 计算 coordinates 数组中第3列和第4列的总和，结果存储在 length 中
    length = coordinates[:, 2:4].sum()
    # 如果 ids 为空，即 coordinates 中没有最后一列为1的行。
    if len(ids) < 1:  # if not exist [0, 0, 1]
        # 查找 coordinates 数组中第4列等于1的索引，并将其值加1，结果存储在 ids 中。
        ids = np.where(coordinates[:, 3] == 1)[0] + 1
        # 如果 ids 仍为空，即 coordinates 中没有第4列为1的行。
        if len(ids) < 1:  # if not exist [0, 1, 0]
            # 将 ids 设置为数组长度。
            ids = np.array([len(coordinates)])
            # 根据 ids 对 coordinates 进行分割，并移除空列表。
            xys_split = np.split(coordinates, ids, axis=0)[:-1]  # remove the blank list
        else:
            xys_split = np.split(coordinates, ids, axis=0)
    else:  # if exist [0, 0, 1]
        remove_end = np.split(coordinates, ids, axis=0)[0]
        ids = np.where(remove_end[:, 3] == 1)[0] + 1  # break in [0, 1, 0]
        xys_split = np.split(remove_end, ids, axis=0)[:-1]  # split from the remove_end

    coord_list = []
    # 笔画（stroke）
    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        if len(xs) > 0:
            xys = np.stack([xs, ys], axis=-1).reshape(-1)
            coord_list.append(xys)
        else:
            pass
    return coord_list, length


def corrds2xys(coordinates):
    """
    description:
        [x, y] --> [x, y, p1, p2, p3]
        see 'A NEURAL REPRESENTATION OF SKETCH DRAWINGS' for more details
    new_strokes = array([
        [x1, y1, 1, 0, 0],
        [x2, y2, 1, 0, 0],
        ...,
        [xn, yn, 0, 1, 0],  # 笔画1的结束点
        [x1, y1, 1, 0, 0],
        ...,
        [xm, ym, 0, 1, 0]   # 笔画2的结束点
    ])
    """
    new_strokes = []
    for stroke in coordinates:
        # 遍历笔画中的每个(x, y)对，将其重塑为(-1, 2)的形状。
        for (x, y) in np.array(stroke).reshape((-1, 2)):
            # 生成一个五维向量 [x, y, 1, 0, 0]，数据类型为 float32。
            p = np.array([x, y, 1, 0, 0], np.float32)
            new_strokes.append(p)
        try:
            # 将每个笔画的最后一个点的后三个元素设置为 [0, 1, 0]，表示笔画的结束。
            new_strokes[-1][2:] = [0, 1, 0]  # set the end of a stroke
        except IndexError:
            print(stroke)
            return None
    # 将 new_strokes 列表转换为NumPy数组。
    new_strokes = np.stack(new_strokes, axis=0)
    return new_strokes


if __name__ == '__main__':
    file_path = '.'
    file_name = 'test.pkl'
    imgs_path = ['../style_samples/1_binary.jpg', '../style_samples/2_binary.jpg', '../style_samples/3_binary.jpg']
    write_pkl(file_path, file_name, imgs_path, 2)
