# -*- coding: utf-8 -*-
def unicode_to_gbk(char):
    return char.encode('gbk').decode('unicode_escape')


def create_gbk_encoding_dict(chars):
    encoding_dict = {}
    for char in chars:
        encoding_dict[char] = [f"\\u{ord(char):04x}"]
    return encoding_dict


all_char = []
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
all_char.extend(chinese_punctuations)
all_char.extend(numbers_str)
all_char.extend(english_str)

if __name__ == '__main__':
    # 转换为GBK编码字典
    gbk_encoding_dict = create_gbk_encoding_dict(all_char)
    # print(gbk_encoding_dict)
    for k, v in gbk_encoding_dict.items():
        # print(k, v[0])
        print("\"" + v[0] + "\",")
