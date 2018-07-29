import numpy as np
import random
from python import Input


def entropy_b2(prob):
    return -prob*np.log2(prob)


# Top, Right, Bottom, Left
_overlays = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def generate_sliding_overlay(dim):
    _overlay = []
    for i in range(1-dim, dim):
        for j in range(1-dim, dim):
            if i is not 0 or j is not 0:
                _overlay.append((i, j))
    return _overlay


class Model:

    def __init__(self, tile_dir, output_shape, dim, rotate_patterns=False, iteration_limit=-1, overlays=_overlays):
        self.tiles = Input.load_tiles(tile_dir)
        self.img_shape = output_shape
        self.dim = dim
        self.rotate_patterns = rotate_patterns
        self.iteration_limit = iteration_limit
        self.overlays = overlays

        self.patterns = []
        self.counts = []
        self.fit_table = []
        self.propagate_stack = []
        self.probs = None

        self.create_waveforms(dim)

        self.num_patterns = len(self.patterns)
        self.wave_shape = (self.img_shape[0]+1 - dim, self.img_shape[1]+1 - dim)

        self.check_fits()

        self.waves = np.full(self.wave_shape + (self.num_patterns,), True)
        self.observed = np.full(self.wave_shape, False)
        self.registered_propogate = np.full(self.wave_shape, False)
        # self.entropies = np.full(self.wave_shape, -np.sum(self.probs * np.log2(self.probs)))
        self.entropies = np.ones(self.wave_shape, dtype=np.int16)*self.num_patterns
        self.out_img = np.full(self.img_shape + (3,), -1.)

        print(self.fit_table.shape)
        print(self.wave_shape)
        print(self.waves.shape)

    def generate_image(self):
        row, col = random.randint(0, self.wave_shape[0]-1), random.randint(0, self.wave_shape[1]-1)
        iteration = 0
        while row >= 0 and col >= 0 and (self.iteration_limit<0 or iteration<self.iteration_limit):
            self.observe_wave(row, col)
            self.propagate()
            row, col = self.get_lowest_entropy()
            iteration += 1
            if iteration % 100 == 0:
                print("iteration: {}".format(iteration))

        for row in range(self.wave_shape[0]):
            for col in range(self.wave_shape[1]):
                self.render_superpositions(row, col)

    def render_superpositions(self, row, col):
        num_valid_patterns = sum(self.waves[row, col])
        self.out_img[row:row+self.dim, col:col+self.dim] = np.zeros((self.dim, self.dim, 3))
        for i in range(self.num_patterns):
            if self.waves[row, col, i]:
                self.out_img[row:row+self.dim, col:col+self.dim] += self.patterns[i] / num_valid_patterns

    def get_lowest_entropy(self):
        lowest_val = -1
        r = -1
        c = -1
        for col in range(self.wave_shape[1]):
            for row in range(self.wave_shape[0]):
                if not self.observed[row, col] and self.waves[row, col].any():
                    if lowest_val < 0 or (lowest_val > self.entropies[row, col] > 0):
                        lowest_val = self.entropies[row, col]
                        r = row
                        c = col
        return r, c

    def observe_wave(self, row, col):
        possible_indices = []
        sub_probs = []
        for i in range(self.num_patterns):
            if self.waves[row, col, i]:
                possible_indices.append(i)
                sub_probs.append(self.counts[i])
        collapsed_index = np.random.choice(possible_indices, p=sub_probs/np.sum(sub_probs))
        self.do_observe(row, col, collapsed_index)
        self.propagate_stack.append((row, col))

    def do_observe(self, row, col, pattern_index):
        self.observed[row, col] = True
        self.waves[row, col] = np.full((self.num_patterns,), False)
        self.waves[row, col, pattern_index] = True
        # self.out_img[row:row+self.dim, col:col+self.dim] = self.patterns[pattern_index]

    def propagate(self):
        iterations = 0
        while len(self.propagate_stack) > 0:
            row, col = self.propagate_stack.pop()
            self.registered_propogate[row, col] = False
            valid_indices = []
            for i in range(self.num_patterns):
                if self.waves[row, col, i]:
                    valid_indices.append(i)

            if valid_indices is None or len(valid_indices) is 0:
                print("Error: contradiction with no valid indices")
                continue

            for overlay_idx in range(len(self.overlays)):
                self.update_wave(row, col, overlay_idx, valid_indices)

            iterations += 1
            if iterations % 1000 == 0:
                print("propagation iteration: {}, propogation stack size: {}".format(iterations, len(self.propagate_stack)))

    def update_wave(self, row, col, overlay_idx, valid_indices):
        col_shift, row_shift = self.overlays[overlay_idx]
        row_s = row+row_shift
        col_s = col+col_shift
        if row_s >= 0 and row_s < self.wave_shape[0] and \
                col_s >= 0 and col_s < self.wave_shape[1] and \
                not self.observed[row_s, col_s]:
            changed = False
            valid_pattern_count = 0
            valid_pattern_idx = -1
            for i in range(self.num_patterns):
                if self.waves[row_s, col_s, i]:
                    can_fit = False
                    j = 0
                    while j < len(valid_indices) and not can_fit:
                        can_fit = self.fit_table[valid_indices[j], i, overlay_idx]
                        j += 1
                    if not can_fit:
                        self.waves[row_s, col_s, i] = False
                        # self.entropies[row_s, col_s] -= entropy_b2(self.probs[i])
                        self.entropies[row_s, col_s] -= 1
                        changed = True
                    else:
                        valid_pattern_count += 1
                        valid_pattern_idx = i
            if valid_pattern_count == 1:
                self.do_observe(row_s, col_s, valid_pattern_idx)
            if changed and not self.registered_propogate[row_s, col_s]:
                self.propagate_stack.append((row_s, col_s))
                self.registered_propogate[row_s, col_s] = True

    def create_waveforms(self, dim):
        height, width, depth = self.tiles[0].shape

        for tile in self.tiles:
            for col in range(width + 1 - dim):
                for row in range(height + 1 - dim):
                    pattern = tile[row:row+dim, col:col+dim]
                    if self.rotate_patterns:
                        for rot in range(4):
                            self.add_waveform(np.rot90(pattern, rot))
                    else:
                        self.add_waveform(pattern)

        self.probs = np.array(self.counts) / sum(self.counts)

    def add_waveform(self, waveform):
        for i in range(len(self.patterns)):
            if np.array_equal(waveform, self.patterns[i]):
                self.counts[i] += 1
                return
        self.patterns.append(waveform)
        self.counts.append(1)

    def check_fits(self):
        self.fit_table = np.full((self.num_patterns, self.num_patterns, len(self.overlays)), False)

        for patt_idx1 in range(self.num_patterns):
            patt1 = self.patterns[patt_idx1]
            for patt_idx2 in range(self.num_patterns):
                patt2 = self.patterns[patt_idx2]

                for i in range(len(self.overlays)):
                    col_shift, row_shift = self.overlays[i]

                    row_start_1 = max(row_shift, 0)
                    row_end_1 = min(row_shift+self.dim-1, self.dim-1)
                    col_start_1 = max(col_shift, 0)
                    col_end_1 = min(col_shift+self.dim-1, self.dim-1)

                    row_start_2 = row_start_1 - row_shift
                    row_end_2 = row_end_1 - row_shift
                    col_start_2 = col_start_1 - col_shift
                    col_end_2 = col_end_1 - col_shift

                    self.fit_table[patt_idx1, patt_idx2, i] = np.array_equal(
                        patt1[row_start_1:row_end_1+1, col_start_1:col_end_1+1],
                        patt2[row_start_2:row_end_2+1, col_start_2:col_end_2+1])
