import struct
import os
import numpy as np
import cv2


# POT文件中每个汉字的头信息
class POT_HEADER:
    def __init__(self, sample_size, tag_code, stroke_number):
        self.sample_size = sample_size
        self.tag_code = tag_code
        self.stroke_number = stroke_number


# 笔画中的点信息
class COORDINATE:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 一个汉字的结束标记
class END_TAG:
    def __init__(self, endflag0, endflag1):
        self.endflag0 = endflag0
        self.endflag1 = endflag1


STROKE_END_X = -1
STROKE_END_Y = 0


def get_gb_from_2char(high, low):
    return (high << 8) + low


def read_a_pot(pot_filepath):
    coords_list = []
    labels = []

    with open(pot_filepath, 'rb') as f:
        while True:
            header_data = f.read(8)
            if len(header_data) < 8:
                break

            sample_size, tag_code_0, tag_code_1, tag_code_2, tag_code_3, stroke_number = struct.unpack('<HBBBBH',
                                                                                                       header_data)
            header = POT_HEADER(sample_size, [tag_code_0, tag_code_1, tag_code_2, tag_code_3], stroke_number)

            strokes = []
            for _ in range(header.stroke_number):
                stroke = []
                while True:
                    coord_data = f.read(4)
                    x, y = struct.unpack('<hh', coord_data)
                    if x == STROKE_END_X and y == STROKE_END_Y:
                        break
                    stroke.append((x, y))
                strokes.append(stroke)

            end_tag_data = f.read(4)
            end_tag = END_TAG(*struct.unpack('<hh', end_tag_data))

            gb_code = get_gb_from_2char(header.tag_code[1], header.tag_code[0])
            coords_list.append(strokes)
            labels.append(gb_code)

    return coords_list, labels


def gb_to_hanzi(gb_code):
    high_byte = (gb_code >> 8) & 0xFF
    low_byte = gb_code & 0xFF
    try:
        return bytes([high_byte, low_byte]).decode('gb2312')
    except UnicodeDecodeError:
        return None


# Example usage
pot_filepath = '006.pot'
coords_list, labels = read_a_pot(pot_filepath)

# Print the results
for i in range(len(coords_list)):
    hanzi = gb_to_hanzi(labels[i])
    if hanzi:
        print(f'Hanzi: {hanzi}')
    else:
        print(f'Hanzi GB code: {labels[i]} (Unable to decode)')
    print(f'Number of strokes: {len(coords_list[i])}')
    for stroke in coords_list[i]:
        print(f'Stroke: {stroke}')
    print()
    break
