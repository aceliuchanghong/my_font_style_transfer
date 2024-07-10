import pickle
import argparse

chinese_punctuations = [
    '刘',
]

english_str = ['A', 'B', ]
char_dict = pickle.load(open('../generate_utils/old_character_dict.pkl', 'rb'))


def main(opt):
    all_char = []
    i = 0
    temp_char = []
    if not opt.not_all:
        all_char = chinese_punctuations + english_str
    while i < opt.nums and i < len(char_dict):
        temp_char.append(char_dict[i])
        i += 1
    all_char.extend(temp_char)
    if len(all_char) < 90:
        print(all_char)
    pickle.dump(all_char, open('../generate_utils/new_character_dict.pkl', 'wb'))
    print("suc generate:", len(all_char))


if __name__ == "__main__":
    """
    cd z_new_start/ABtest
    python 2_gen_ABcharacter_pkl.py --nums 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_all', action='store_true', help='有not_all参数的时候不加字母标点之类')
    parser.add_argument('--nums', default=10000, type=int, help='选择多少字生成')
    opt = parser.parse_args()
    main(opt)
