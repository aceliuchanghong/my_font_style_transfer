import pickle
import argparse


def main(opt):
    char_dict = pickle.load(open(opt.pkl, 'rb'))
    print(char_dict)
    return char_dict


if __name__ == '__main__':
    """
    conda activate SDTLog1
    cd z_new_start/generate_utils
    python read_character_pkl.py
    python read_character_pkl.py --pkl ../ABtest/files/new_character_dict.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='new_character_dict.pkl', help='读取文件')
    opt = parser.parse_args()
    main(opt)
