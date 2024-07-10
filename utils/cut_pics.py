import argparse
import os
import cv2
import shutil


def main(opt):
    # # 膨胀腐蚀大小 即将笔画膨胀 避免将偏旁单独分割 根据图片请况自行设置
    # rect_size = 8
    # # 字体小于该值忽略 20*20
    # ignore_min_size = 80
    # # 字体大于该值忽略 100*100
    # ignore_max_size = 150
    # # 图片选取偏移
    # offset_param = 48

    rect_size = opt.rect_size
    ignore_min_size = opt.ignore_min_size
    ignore_max_size = opt.ignore_max_size
    offset_param = opt.offset_param

    # 需要切分的图片 input/input.jpg
    input_file = opt.input_path

    # 输出文件夹 output
    output_path = opt.output_path + os.path.sep
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    img = cv2.imread(input_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化 根据图片情况调节灰度阈值 第二个参数灰度值表示小于该值就将其变为0
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path + "thresh.jpg", thresh)
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # 待切分的图片 img原图 gray灰度图 thresh二值化后的灰度图 thresh_rgb二值化后转RGB
    result = thresh.copy()

    # 膨胀腐蚀 将字体笔画扩大15像素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_size, rect_size))
    eroded = cv2.erode(thresh, kernel)
    cv2.imwrite(output_path + "eroded.jpg", eroded)

    # 轮廓检测
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 膨胀间距
    spacing = rect_size // 2 - 1
    cut_frame_color = (0, 255, 0)
    font_frame_color = (255, 0, 0)

    token = 1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < ignore_min_size or h < ignore_min_size or w > ignore_max_size or h > ignore_max_size:
            continue

        font_w = w - spacing - spacing
        font_h = h - spacing - spacing

        frame_size = font_w if font_w > font_h else font_h

        # 正方形切割偏移量
        x_offset = int((frame_size - font_w) / 2)
        y_offset = int((frame_size - font_h) / 2)

        start_x = x + spacing - x_offset - offset_param
        start_y = y + spacing - y_offset - offset_param
        end_x = x + w - spacing + x_offset + offset_param
        end_y = y + h - spacing + y_offset + offset_param

        # 字体框线
        cv2.rectangle(thresh_rgb, (x + spacing, y + spacing), (x + w - spacing, y + h - spacing), font_frame_color, 1)
        # 切割框线
        cv2.rectangle(thresh_rgb, (start_x, start_y), (end_x, end_y), cut_frame_color, 1)
        # 切割
        temp = result[start_y:end_y, start_x:end_x]
        path = opt.output_path + os.path.sep + "result" + os.path.sep
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + str(token) + "_" + str(x) + "_" + str(y) + ".jpg", temp)

        token += 1

    cv2.imwrite(output_path + "result.jpg", thresh_rgb)


if __name__ == '__main__':
    """Parse input arguments
    python cut_pics.py --input 'from/from.jpg' --rect_size 8 --ignore_min_size 80 --ignore_max_size 150 --offset_param 3
    python cut_pics.py --input 'from/from1.jpg' --rect_size 8 --ignore_min_size 50 --ignore_max_size 70 --offset_param 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input_path', default='from/from4.jpg',
                        help='Please set the input path')
    parser.add_argument('--output', dest='output_path', default='font',
                        help='Please set the output path')
    parser.add_argument('--rect_size', dest='rect_size', default=8, type=int,
                        help='膨胀腐蚀大小')
    parser.add_argument('--ignore_min_size', dest='ignore_min_size', default=80, type=int,
                        help='字体小于该值忽略')
    parser.add_argument('--ignore_max_size', dest='ignore_max_size', default=150, type=int,
                        help='字体大于该值忽略')
    parser.add_argument('--offset_param', dest='offset_param', default=48, type=int,
                        help='图片选取偏移,选取图片扩大范围')
    opt = parser.parse_args()
    main(opt)
