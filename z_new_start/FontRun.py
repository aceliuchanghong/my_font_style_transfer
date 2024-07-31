import math
import random
import pickle


class CoorsSubject:
    def __init__(self):
        self.output = {}
        self.personality = {}
        self.scale_factor = 1

    def set_scale(self, scale_factor):
        self.scale_factor = scale_factor

    def generate_personality(self, angle_range=(-math.pi / 100, math.pi / 100)):
        self.personality = {
            'angle': random.uniform(*angle_range),
            'wave_amplitude': random.uniform(3.1, 4.2),
            'wave_frequency': random.uniform(0.35, 0.51),
        }

    def apply_personality(self, x, y):
        nx = x * math.cos(self.personality['angle']) - y * math.sin(self.personality['angle'])
        ny = x * math.sin(self.personality['angle']) + y * math.cos(self.personality['angle'])
        # Apply wave transformation
        nx += self.personality['wave_amplitude'] * math.sin(y * self.personality['wave_frequency'])
        ny += self.personality['wave_amplitude'] * math.sin(x * self.personality['wave_frequency'])
        return nx * self.scale_factor, ny * self.scale_factor

    def disturb_stroke(self, stroke):
        disturbed_stroke = []
        for i in range(len(stroke)):
            x1, y1, p1, p2 = stroke[i]
            x1, y1 = self.apply_personality(x1, y1)
            disturbed_stroke.append((x1, y1, p1, p2))
        return disturbed_stroke

    def request(self, x: dict, path: str) -> str:
        self.generate_personality()

        self.output = {char: [self.disturb_stroke(stroke) for stroke in strokes] for char, strokes in x.items()}

        with open(path, 'wb') as f:
            pickle.dump(self.output, f)
        return path
