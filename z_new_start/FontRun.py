import random
import math
import pickle


class CoorsSubject:
    def __init__(self):
        self.output = {}
        self.global_personality = None

    def generate_global_personality(self):
        return {
            'stroke_spacing': random.uniform(0.7, 0.78),
            'stroke_length': random.uniform(0.99, 0.998),
            'wave_amplitude': random.uniform(4.1, 4.2),
            'wave_frequency': random.uniform(0.4, 0.5),
            'decoration_probability': random.uniform(0.6, 0.7),
            'decoration_size': random.uniform(12, 15),
            'tightness': random.uniform(1.02, 1.24),
            'merge_probability': random.uniform(0.3, 0.5),
            'merge_distance': random.uniform(5, 100),
        }

    def transform_point(self, x, y, personality):
        new_x = x + personality['wave_amplitude'] * math.sin(y * personality['wave_frequency'])
        new_y = y + personality['wave_amplitude'] * math.sin(x * personality['wave_frequency'])
        return new_x, new_y

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
                stroke.append((x, y, 0, 1))
            elif decoration_type == 'tail':
                tail_x = x + personality['decoration_size'] * 1.5
                tail_y = y
                stroke.append((tail_x, tail_y, 0, 0))
                stroke.append((x, y, 0, 1))
            else:
                dot_x = x + random.uniform(-1, 1) * personality['decoration_size'] / 2
                dot_y = y + random.uniform(-1, 1) * personality['decoration_size'] / 2
                stroke.append((dot_x, dot_y, 0, 0))
                stroke.append((x, y, 0, 1))
        return stroke

    def smooth_stroke(self, stroke, personality):
        new_stroke = []
        for i, point in enumerate(stroke):
            x, y, p1, p2 = point
            new_x, new_y = self.transform_point(x, y, personality)
            if i > 0:
                prev_x, prev_y = new_stroke[-1][:2]
                dx = new_x - prev_x
                dy = new_y - prev_y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance > 0:
                    new_x = prev_x + dx * personality['stroke_spacing']
                    new_y = prev_y + dy * personality['stroke_spacing']
            new_stroke.append((new_x, new_y, p1, p2))
        if new_stroke[0][:2] != new_stroke[-1][:2]:
            new_stroke.append((new_stroke[0][0], new_stroke[0][1], 0, 1))
        new_stroke = self.add_decoration(new_stroke, personality)
        return new_stroke

    def merge_strokes(self, strokes, personality):
        merged_strokes = []
        i = 0
        while i < len(strokes):
            current_stroke = strokes[i]
            if i + 1 < len(strokes) and random.random() < personality['merge_probability']:
                next_stroke = strokes[i + 1]
                end_point = current_stroke[-1]
                start_point = next_stroke[0]
                distance = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
                if distance < personality['merge_distance']:
                    merged_stroke = current_stroke[:-1] + next_stroke
                    merged_stroke[0] = (merged_stroke[0][0], merged_stroke[0][1], 1, 0)
                    merged_stroke[-1] = (merged_stroke[-1][0], merged_stroke[-1][1], 0, 1)
                    merged_strokes.append(merged_stroke)
                    i += 2
                else:
                    merged_strokes.append(current_stroke)
                    i += 1
            else:
                merged_strokes.append(current_stroke)
                i += 1
        return merged_strokes

    def request(self, x: dict, path: str) -> str:
        self.global_personality = self.generate_global_personality()
        for char, strokes in x.items():
            smoothed_strokes = [self.smooth_stroke(stroke, self.global_personality) for stroke in strokes]
            merged_strokes = self.merge_strokes(smoothed_strokes, self.global_personality)
            self.output[char] = merged_strokes
        with open(path, 'wb') as f:
            pickle.dump(self.output, f)
        return path
