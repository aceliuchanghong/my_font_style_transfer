import pickle
import argparse

chinese_punctuations = [
    '。',  # 句号
    '，',  # 逗号
    '、',  # 顿号
    '；',  # 分号
    '：',  # 冒号
    '？',  # 问号
    '！',  # 感叹号
    '“',  # 左双引号
    '”',  # 右双引号
    '‘',  # 左单引号
    '’',  # 右单引号
    '（',  # 左括号
    '）',  # 右括号
]  # 13
numbers_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

english_str = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_dict = pickle.load(open('old_character_dict.pkl', 'rb'))


def main(opt):
    all_char = []
    i = 0
    temp_char = []
    if not opt.not_all:
        all_char = chinese_punctuations + numbers_str + english_str
    while i < opt.nums and i < len(char_dict):
        temp_char.append(char_dict[i])
        i += 1
    all_char.extend(temp_char)
    if len(all_char) < 90:
        print(all_char)
    pickle.dump(all_char, open('new_character_dict.pkl', 'wb'))
    print("suc generate:", len(all_char))


if __name__ == "__main__":
    """
    conda activate SDTLog1
    cd z_new_start/generate_utils
    python gen_character_pkl.py
    python gen_character_pkl.py --nums 5
    python gen_character_pkl.py --nums 5 --not_all
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_all', action='store_true', help='有not_all参数的时候不加字母标点之类')
    parser.add_argument('--nums', default=10000, type=int, help='选择多少字生成')
    opt = parser.parse_args()
    main(opt)
