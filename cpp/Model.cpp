#include "Model.h"

Model::Model(const std::string tile_dir, Pair &output_shape, const char dim, std::vector<Pair> &overlays, const bool rotate_patterns, const int iteration_limit) : 
dim(dim), iteration_limit(iteration_limit), overlay_count(overlays.size()), img_shape(output_shape), overlays(overlays) {
	load_tiles(tile_dir, tiles);
	create_waveforms(rotate_patterns);

	srand(time(nullptr));

	num_patterns = patterns.size();
	num_patt_2d = Pair(num_patterns, num_patterns);
	wave_shape = Pair(img_shape.x + 1 - dim, img_shape.y + 1 - dim);

	entropies = std::vector<int>(wave_shape.size, num_patterns);
	waves = std::vector<char>(wave_shape.size * num_patterns, true);
	observed = std::vector<char>(wave_shape.size, false);
	registered_propagate = std::vector<char>(wave_shape.size, false);

	out_img = cv::Mat(img_shape.x, img_shape.y, tiles[0].type());

	generate_fit_table();
	std::cout << "Input Tiles: " << tiles.size() << std::endl;
	std::cout << "Patterns: " << num_patterns << std::endl;
	std::cout << "Overlay Count: " << overlay_count << std::endl;
	std::cout << "Wave Shape: " << wave_shape.y << " x " << wave_shape.x << std::endl;
}

void Model::generate_image() {
	std::cout << "Called Generate" << std::endl;
	auto lowest_entropy_idx = Pair(rand_int(wave_shape.x), rand_int(wave_shape.y));
	auto iteration = 0;
	while ((iteration_limit < 0 || iteration < iteration_limit) && lowest_entropy_idx.non_negative()) {
		observe_wave(lowest_entropy_idx);
		propagate();
		get_lowest_entropy(lowest_entropy_idx);

		iteration += 1;
		if (iteration % 100 == 0)
			std::cout << "iteration: " << iteration << std::endl;
	}
	std::cout << "Finished Algorithm" << std::endl;

	for (auto row = 0; row < wave_shape.y; row++) {
		for (auto col = 0; col < wave_shape.x; col++) {
			render_superpositions(row, col);
		}
	}
	std::cout << "Finished Rendering" << std::endl;
}

cv::Mat& Model::get_image() {
	return out_img;
}

void Model::get_lowest_entropy(Pair &idx) {
	auto r = -1; auto c = -1;
	auto lowest_entropy = -1;
	const size_t waves_count = wave_shape.size;
	for (auto wave_idx = 1; wave_idx < waves_count; wave_idx++) {
		const auto entropy_val = entropies[wave_idx];
		if (!observed[wave_idx] && (lowest_entropy < 0 || entropy_val < lowest_entropy && entropy_val > 1)) {
			lowest_entropy = entropy_val;
			r = wave_idx/wave_shape.x; c = wave_idx%wave_shape.x;
		}
	}
	idx.x = c; idx.y = r;
}

void Model::render_superpositions(const int row, const int col) {
	const int idx_row_col_patt_base = row*wave_shape.x*num_patterns + col*num_patterns;
	auto num_valid_patterns = 0;
	std::vector<int> valid_patt_idxs;
	for (auto patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
		if (waves[idx_row_col_patt_base + patt_idx]) {
			num_valid_patterns += 1;
			valid_patt_idxs.push_back(patt_idx);
		}
	}
	for (auto r = row; r < row + dim; r++) {
		for (auto c = col; c < col + dim; c++) {
			auto& bgr = out_img.ptr<BGR>(r)[c];
			if (num_valid_patterns > 0) {
				bgr = BGR(0, 0, 0);
				for (auto patt_idx: valid_patt_idxs)
					bgr += patterns[patt_idx].ptr<BGR>(r - row)[c - col] / num_valid_patterns;
			} else {
				bgr = BGR(204, 51, 255);
			}
		}
	}
}

void Model::observe_wave(Pair &wavef) {
	const auto idx_row_col_patt_base = get_idx(wavef, wave_shape, num_patterns, 0);
	auto possible_patterns_sum = 0;
	std::vector<int> possibleIndices;
	for (auto i = 0; i < num_patterns; i++) {
		if (waves[idx_row_col_patt_base + i]) {
			possible_patterns_sum += counts[i];
			possibleIndices.push_back(i);
		}
	}

	auto rnd = rand_int(possible_patterns_sum)+1;
	int collapsed_index;

	for (collapsed_index = 0; collapsed_index < num_patterns && rnd>0; collapsed_index++) {
		if (waves[idx_row_col_patt_base + collapsed_index]) {
			rnd -= counts[collapsed_index];
		}
	}
	collapsed_index -= 1;	// Counter-action against additional increment from for-loop

	observed[get_idx(wavef, wave_shape, 1, 0)] = true;
	for (auto patt_idx = 0; patt_idx < num_patterns; patt_idx++)
		waves[idx_row_col_patt_base + patt_idx] = (patt_idx == collapsed_index);

	register_waveform(wavef);
}

void Model::propagate() {
	auto iterations = 0;
	while (!propagate_stack.empty()) {
		auto wavef = pop_waveform();
		const auto idx_row_col_patt_base = get_idx(wavef, wave_shape, num_patterns, 0);
		std::vector<int> valid_indices;
		for (auto i = 0; i < num_patterns; i++) {
			if (waves[idx_row_col_patt_base + i])
				valid_indices.push_back(i);
		}

		if (valid_indices.empty()) {
			std::cout << "Error: contradiction with no valid indices!" << std::endl;
			continue;
		}

		for (auto overlay_idx = 0; overlay_idx < overlay_count; overlay_idx++)
			update_wave(wavef, overlay_idx, valid_indices);

		iterations += 1;
		if (iterations % 1000 == 0)
			std::cout << "propagation iteration: " << iterations << ", propagation stack size: " << propagate_stack.size() << std::endl;
	}
}

void Model::register_waveform(Pair& waveform) {
	const auto idx = waveform.y * wave_shape.x + waveform.x;
	registered_propagate[idx] = true;
	propagate_stack.push(waveform);
}

Pair Model::pop_waveform() {
	const auto out = propagate_stack.top();
	const auto idx = out.y * wave_shape.x + out.x;
	registered_propagate[idx] = false;
	propagate_stack.pop();
	return out;
}

void Model::update_wave(Pair &wavef, const int overlay_idx, std::vector<int> &valid_indices) {
	auto idx_s = wavef + overlays[overlay_idx];
	const auto idx_row_col = get_idx(idx_s, wave_shape, 1, 0);
	if (idx_s.non_negative() && idx_s < wave_shape && !observed[get_idx(idx_s, wave_shape, 1, 0)]) {
		const auto idx_row_col_patt_base = get_idx(idx_s, wave_shape, num_patterns, 0);

		const auto valid_indices_count = valid_indices.size();
		const auto entropy = &entropies[idx_row_col];

		auto changed = false;
		auto valid_pattern_count = 0;
		for (auto overlay_patt_idx = 0; overlay_patt_idx < num_patterns; overlay_patt_idx++) {
			if (waves[idx_row_col_patt_base + overlay_patt_idx]) {
				auto can_fit = false;
				for (auto center_patt_idx = 0; center_patt_idx < valid_indices_count && !can_fit; center_patt_idx++) {
					Pair lookup_idx(overlay_patt_idx, valid_indices[center_patt_idx]);
					can_fit = fit_table[get_idx(lookup_idx, num_patt_2d, overlay_count, overlay_idx)];
				}

				if (!can_fit) {
					waves[idx_row_col_patt_base + overlay_patt_idx] = false;
					*entropy -= 1;
					changed = true;
				} else {
					valid_pattern_count += 1;
				}
			}
		}

		if (valid_pattern_count <= 1) {		// Counts for both collapsed waves and contradictions
			observed[idx_row_col] = true;
			if (changed && !registered_propagate[idx_row_col]) {
				register_waveform(idx_s);
			}
		}
	}
}

void Model::create_waveforms(const bool rotate_patterns) {
	const auto height = tiles[0].cols;
	const auto width = tiles[0].rows;
	auto channels = tiles[0].channels();

	for (const auto& tile : tiles) {
		for (auto col = 0; col < width + 1 - dim; col++) {
			for (auto row = 0; row < height + 1 - dim; row++) {
				auto pattern = tile(cv::Rect(col, row, dim, dim));
				add_waveform(pattern);
				if (rotate_patterns) {
					cv::Mat cpy;
					cv::rotate(pattern, cpy, cv::ROTATE_90_COUNTERCLOCKWISE);
					add_waveform(cpy);
					cv::Mat cpy1;
					cv::rotate(pattern, cpy1, cv::ROTATE_180);
					add_waveform(cpy1);
					cv::Mat cpy2;
					cv::rotate(pattern, cpy2, cv::ROTATE_90_CLOCKWISE);
					add_waveform(cpy2);
				}
			}
		}
	}
}

void Model::add_waveform(const cv::Mat &waveform) {
	const auto curr_patt_count = patterns.size();
	for (auto i = 0; i < curr_patt_count; i++) {
		if (patterns_equal(waveform, patterns[i])) {
			counts[i] += 1;
			return;
		}
	}
	patterns.push_back(waveform);
	counts.push_back(1);
}

void Model::generate_fit_table() {
	for (auto patt1 : patterns) {
		for (auto patt2 : patterns) {
			for (auto overlay : overlays) {
				fit_table.push_back(overlay_fit(patt1, patt2, overlay, dim));
			}
		}
	}
}
