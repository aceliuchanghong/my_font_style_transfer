import random
import math
import pickle

class CoorsSubject:
    def __init__(self):
        self.output = {}
        self.global_personality = None

    def generate_global_personality(self):
        return {
            'stroke_spacing': random.uniform(0.85, 0.9),
            'wave_amplitude': random.uniform(14.1, 15.2),
            'wave_frequency': random.uniform(0.25, 0.4),
            'horizontal_shift': random.uniform(48, 50),
            'vertical_shift': random.uniform(48, 50),
            'shift_probability': random.uniform(0.69, 0.7),
        }

    def transform_point(self, x, y, personality):
        new_x = x + personality['wave_amplitude'] * math.sin(y * personality['wave_frequency'])
        new_y = y + personality['wave_amplitude'] * math.sin(x * personality['wave_frequency'])
        return new_x, new_y

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
        return new_stroke

    def round_corners(self, stroke):
        rounded_stroke = []
        for i, point in enumerate(stroke):
            x, y, p1, p2 = point
            if i == 0 or i == len(stroke) - 1:
                rounded_stroke.append(point)
            else:
                prev_point = stroke[i - 1]
                next_point = stroke[i + 1]
                new_x = (prev_point[0] + x + next_point[0]) / 3
                new_y = (prev_point[1] + y + next_point[1]) / 3
                rounded_stroke.append((new_x, new_y, p1, p2))
        return rounded_stroke

    def identify_stroke_type(self, stroke):
        x_coords = [point[0] for point in stroke]
        y_coords = [point[1] for point in stroke]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        return 'horizontal' if x_range > y_range else 'vertical'

    def shift_stroke(self, stroke, personality):
        if random.random() < personality['shift_probability']:
            stroke_type = self.identify_stroke_type(stroke)
            shift = personality['horizontal_shift'] if stroke_type == 'horizontal' else personality['vertical_shift']
            shifted_stroke = []
            for point in stroke:
                x, y, p1, p2 = point
                if stroke_type == 'horizontal':
                    shifted_stroke.append((x, y + shift, p1, p2))
                else:
                    shifted_stroke.append((x + shift, y, p1, p2))
            return shifted_stroke
        return stroke

    def process_character(self, strokes, personality):
        smoothed_strokes = [self.smooth_stroke(stroke, personality) for stroke in strokes]
        rounded_strokes = [self.round_corners(stroke) for stroke in smoothed_strokes]
        shifted_strokes = [self.shift_stroke(stroke, personality) for stroke in rounded_strokes]
        return shifted_strokes

    def request(self, x: dict, path: str) -> str:
        self.global_personality = self.generate_global_personality()
        for char, strokes in x.items():
            self.output[char] = self.process_character(strokes, self.global_personality)
        with open(path, 'wb') as f:
            pickle.dump(self.output, f)
        return path
