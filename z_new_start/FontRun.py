import math
import random
import pickle

class CoorsSubject:
    def __init__(self):
        self.output = {}
        self.personality = {}

    def generate_personality(self, angle_range=(-math.pi / 90, math.pi / 90)):
        self.personality = {
            'angle': random.uniform(*angle_range),
            'wave_amplitude': random.uniform(4, 5),
            'wave_frequency': random.uniform(0.35, 0.4),
        }

    def apply_personality(self, x, y):
        # Apply rotation based on the angle in personality
        nx = x * math.cos(self.personality['angle']) - y * math.sin(self.personality['angle'])
        ny = x * math.sin(self.personality['angle']) + y * math.cos(self.personality['angle'])
        return nx, ny

    def disturb_coordinate(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return x1, y1
        dx /= length
        dy /= length

        tangent_x, tangent_y = dx, dy
        normal_x, normal_y = -dy, dx

        disturb_x = random.gauss(0, 0.08) * normal_x
        disturb_y = random.gauss(0, 0.08) * normal_y

        return self.apply_personality(x1 + disturb_x, y1 + disturb_y)

    def transform_point_with_wave(self, x, y):
        # Apply wave transformation
        wave_x = x + self.personality['wave_amplitude'] * math.sin(y * self.personality['wave_frequency'])
        wave_y = y + self.personality['wave_amplitude'] * math.sin(x * self.personality['wave_frequency'])
        return wave_x, wave_y

    def disturb_stroke(self, stroke):
        disturbed_stroke = []
        for i in range(len(stroke)):
            x1, y1, p1, p2 = stroke[i]
            if i == len(stroke) - 1:
                x2, y2 = stroke[0][0], stroke[0][1]
            else:
                x2, y2 = stroke[i + 1][0], stroke[i + 1][1]

            # Apply disturbance to coordinate
            x1, y1 = self.disturb_coordinate(x1, y1, x2, y2)

            # Apply wave transformation
            x1, y1 = self.transform_point_with_wave(x1, y1)

            disturbed_stroke.append((x1, y1, p1, p2))
        return disturbed_stroke

    def request(self, x: dict, path: str) -> str:
        self.generate_personality()

        self.output = {char: [self.disturb_stroke(stroke) for stroke in strokes] for char, strokes in x.items()}

        with open(path, 'wb') as f:
            pickle.dump(self.output, f)
        return path
