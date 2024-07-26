import random
import math
import pickle

class CoorsSubject:
    def __init__(self):
        self.output = {}
        self.global_personality = None

    def generate_global_personality(self):
        return {
            'stroke_spacing': random.uniform(0.7, 0.78),  # 控制笔画间距
            'stroke_length': random.uniform(0.99, 0.998),  # 控制笔画长度
            'wave_amplitude': random.uniform(3.1, 3.2),  # 控制波浪幅度
            'wave_frequency': random.uniform(0.3, 0.4),  # 控制波浪频率
            'decoration_probability': random.uniform(0.5, 0.6),  # 控制装饰元素出现的概率
            'decoration_size': random.uniform(19, 20),  # 控制装饰元素大小
            'stroke_shift': random.uniform(-10, 10),  # 控制笔画整体偏移
            'stroke_rotate_angle': random.uniform(-math.pi / 36, math.pi / 36),  # 控制笔画微小旋转角度
            'stroke_spacing_factor': random.uniform(0.8, 1.2),  # 控制笔画间距紧凑程度
        }

    def transform_point(self, x, y, center_x, center_y, personality, is_start=False, is_end=False):
        #笔画进行微小旋转
        new_x = (x - center_x) * math.cos(personality['stroke_rotate_angle']) - (y - center_y) * math.sin(personality['stroke_rotate_angle']) + center_x
        new_y = (x - center_x) * math.sin(personality['stroke_rotate_angle']) + (y - center_y) * math.cos(personality['stroke_rotate_angle']) + center_y

        # 以下步骤保持不变
        new_x = new_x + personality['wave_amplitude'] * math.sin(new_y * personality['wave_frequency'])
        new_y = new_y + personality['wave_amplitude'] * math.sin(new_x * personality['wave_frequency'])

        if is_end:
            new_x = center_x + (new_x - center_x) * personality['stroke_length']
            new_y = center_y + (new_y - center_y) * personality['stroke_length']

        new_x += personality['stroke_shift']
        new_y += personality['stroke_shift']

        return new_x, new_y

    def smooth_stroke(self, stroke, personality):
        new_stroke = []
        center_x = sum(p[0] for p in stroke) / len(stroke)
        center_y = sum(p[1] for p in stroke) / len(stroke)

        for i, point in enumerate(stroke):
            x, y, p1, p2 = point
            is_start = (i == 0)
            is_end = (i == len(stroke) - 1)
            new_x, new_y = self.transform_point(x, y, center_x, center_y, personality, is_start, is_end)

            if i > 0:
                prev_x, prev_y = new_stroke[-1][:2]
                dx = new_x - prev_x
                dy = new_y - prev_y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance > 0:
                    #根据设置的紧凑程度调整新的位置
                    new_x = prev_x + dx * personality['stroke_spacing'] * personality['stroke_spacing_factor']
                    new_y = prev_y + dy * personality['stroke_spacing'] * personality['stroke_spacing_factor']

            new_stroke.append((new_x, new_y, p1, p2))

        # 确保笔画封闭
        if new_stroke[0][:2]!= new_stroke[-1][:2]:
            new_stroke.append((new_stroke[0][0], new_stroke[0][1], 0, 1))

        new_stroke = self.add_decoration(new_stroke, personality)

        return new_stroke

    def add_decoration(self, stroke, personality):
        if random.random() < personality['decoration_probability']:
            last_point = stroke[-1]
            x, y, _, _ = last_point
            decoration_type = random.choice(['hook', 'tail', 'dot'])

            if decoration_type == 'hook':
                hook_x = x + personality['decoration_size']
                hook_y = y - personality['decoration_size']
                stroke.append((hook_x, hook_y, 0, 0))
                stroke.append(
                    (hook_x + personality['decoration_size'] / 2, hook_y + personality['decoration_size'] / 2, 0, 0))
                stroke.append((x, y, 0, 1))  # 回到起点
            elif decoration_type == 'tail':
                tail_x = x + personality['decoration_size'] * 1.5
                tail_y = y
                stroke.append((tail_x, tail_y, 0, 0))
                stroke.append((x, y, 0, 1))  # 回到起点
            else:
                dot_x = x + random.uniform(-1, 1) * personality['decoration_size'] / 2
                dot_y = y + random.uniform(-1, 1) * personality['decoration_size'] / 2
                stroke.append((dot_x, dot_y, 0, 0))
                stroke.append((x, y, 0, 1))  # 回到起点

        return stroke

    def request(self, x: dict, path: str) -> str:
        self.global_personality = self.generate_global_personality()

        for char, strokes in x.items():
            self.output[char] = [self.smooth_stroke(stroke, self.global_personality) for stroke in strokes]

        with open(path, 'wb') as f:
            pickle.dump(self.output, f)
        return path
