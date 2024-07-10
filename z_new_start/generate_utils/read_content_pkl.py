import pickle
import matplotlib.pyplot as plt
import argparse


def main(opt):
    test_style_samples01 = pickle.load(open(opt.pkl, 'rb'))
    show_num_img = opt.nums
    for i, item in enumerate(test_style_samples01):
        print(item)
        plt.imshow(item['img'], cmap='gray')
        plt.show()
        if i + 1 >= show_num_img:
            break


if __name__ == '__main__':
    """
    conda activate SDTLog1
    cd z_new_start/generate_utils
    python read_content_pkl.py
    python read_content_pkl.py --pkl lch_pics_pkl/HYWeiBeiJ.pkl --nums 2
    python read_content_pkl.py --pkl ../ABtest/files/AB_pics_pkl/HYBaManTiW.pkl --nums 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='new_chinese_content.pkl', help='读取文件')
    parser.add_argument('--nums', default=2, type=int, help='展示数量')
    opt = parser.parse_args()
    main(opt)
