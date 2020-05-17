#include "Model.h"
#include "Input.h"
#include "WFCUtil.h"

Model::Model(pair &output_shape, std::vector<cv::Mat> &templates, const char dim, 
	std::vector<pair> &overlays, const bool rotate_patterns, const bool periodic, 
	const int iteration_limit) :
templates(templates), dim(dim), periodic_(periodic), iteration_limit(iteration_limit), overlay_count(overlays.size()),
img_shape(output_shape), overlays(overlays) {
	create_waveforms(rotate_patterns);
	srand(time(nullptr));

	// Initialize size fields
	num_patterns = patterns.size();
	num_patt_2d = pair(num_patterns, num_patterns);
	wave_shape = pair(img_shape.x + 1 - dim, img_shape.y + 1 - dim);

	// Initialize collections.
	propagate_stack_ = std::vector<waveform>(wave_shape.size * num_patterns);
	entropy_ = std::vector<int>(wave_shape.size);
	waves_ = std::vector<char>(wave_shape.size * num_patterns);
	observed_ = std::vector<int>(wave_shape.size);
	compatible_neighbors_ = std::vector<int>(wave_shape.size * num_patterns * overlay_count);

	// Initialize blank output image
	out_img = cv::Mat(img_shape.x, img_shape.y, templates[0].type());

	generate_fit_table();
	std::cout << "Input Templates: " << templates.size() << std::endl;
	std::cout << "Patterns: " << num_patterns << std::endl;
	std::cout << "Overlay Count: " << overlay_count << std::endl;
	std::cout << "Wave Shape: " << wave_shape.y << " x " << wave_shape.x << std::endl;
}

void Model::generate_image() {
	std::cout << "Called Generate" << std::endl;

	// Initialize board into complete superposition, and pick a random wave to collapse
	clear();
	pair lowest_entropy_idx = pair(rand_int(wave_shape.x), rand_int(wave_shape.y));
	int iteration = 0;
	while ((iteration_limit < 0 || iteration < iteration_limit) && lowest_entropy_idx.non_negative()) {

		/* Standard WFC Loop:
		 *		1. Observe a wave and collapse it's state
		 *		2. Propagate the changes throughout the board and update superposition
		 *		   to only allowed states
		 *		3. After the board state has stabilized, find the position of
		 *		   lowest entropy (most likely to be observed) for the next
		 *		   observation.
		 */
		observe_wave(lowest_entropy_idx);
		propagate();
		get_lowest_entropy(lowest_entropy_idx);

		iteration += 1;
		if (iteration % 1000 == 0)
			std::cout << "iteration: " << iteration << std::endl;
	}
	std::cout << "Finished Algorithm" << std::endl;

	// After full collapse (or iteration limit), render the tiles into the output image.
	for (size_t row = 0; row < wave_shape.y; row++) {
		for (size_t col = 0; col < wave_shape.x; col++) {
			render_superpositions(row, col);
		}
	}
	std::cout << "Finished Rendering" << std::endl;
}

cv::Mat& Model::get_image() {
	return out_img;
}

void Model::get_lowest_entropy(pair &idx) {
	int r = -1; int c = -1;
	int lowest_entropy = -1;
	const size_t waves_count = wave_shape.size;

	// Checks all non-collapsed positions to find the position of lowest entropy.
	for (size_t wave_idx = 1; wave_idx < waves_count; wave_idx++) {
		const int entropy_val = entropy_[wave_idx];
		if ((lowest_entropy < 0 || entropy_val < lowest_entropy) && entropy_val > 0 && observed_[wave_idx] == -1) {
			lowest_entropy = entropy_val;
			r = wave_idx/wave_shape.x; c = wave_idx%wave_shape.x;
		}
	}
	idx.x = c;; idx.y = r;
}

void Model::render_superpositions(const int row, const int col) {
	const int idx_row_col_patt_base = row*wave_shape.x*num_patterns + col*num_patterns;

	// Determines the superposition of patterns at this position.
	int num_valid_patterns = 0;
	std::vector<int> valid_patt_idxs;
	for (size_t patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
		if (waves_[idx_row_col_patt_base + patt_idx]) {
			num_valid_patterns += 1;
			valid_patt_idxs.push_back(patt_idx);
		}
	}

	// Renders the "superposition image" for this tile (average of all superimposed states).
	for (int r = row; r < row + dim; r++) {
		for (int c = col; c < col + dim; c++) {
			BGR& bgr = out_img.ptr<BGR>(r)[c];
			if (num_valid_patterns > 0) {
				bgr = BGR(0, 0, 0);
				for (int patt_idx: valid_patt_idxs)
					bgr += patterns[patt_idx].ptr<BGR>(r - row)[c - col] / num_valid_patterns;
			} else {
				// Error: No valid patterns (magenta). Generating a board that is
				// guaranteed to have no errors is NP.
				bgr = BGR(204, 51, 255);
			}
		}
	}
}

void Model::observe_wave(pair &pos) {
	const int idx_row_col_patt_base = get_idx(pos, wave_shape, num_patterns, 0);

	// Determines superposition of states and their total frequency counts.
	int possible_patterns_sum = 0;
	std::vector<int> possible_indices;
	for (size_t i = 0; i < num_patterns; i++) {
		if (waves_[idx_row_col_patt_base + i]) {
			possible_patterns_sum += counts_[i];
			possible_indices.push_back(i);
		}
	}

	int rnd = rand_int(possible_patterns_sum)+1;
	int collapsed_index;

	// Randomly selects a state for collapse. Weighted by state frequency count.
	for (collapsed_index = 0; collapsed_index < num_patterns && rnd>0; collapsed_index++) {
		if (waves_[idx_row_col_patt_base + collapsed_index]) {
			rnd -= counts_[collapsed_index];
		}
	}
	collapsed_index -= 1;	// Counter-action against additional increment from for-loop

	// Bans all other states, since we have collapsed to a single state.
	for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
		if (waves_[idx_row_col_patt_base + patt_idx] != (patt_idx == collapsed_index)) 
			ban_waveform(waveform(pos, patt_idx));
	}

	// Assigns the final state of this position.
	observed_[get_idx(pos, wave_shape, 1, 0)] = collapsed_index;
}

void Model::propagate() {
	int iterations = 0;
	while (stack_index_ > 0) {
		const waveform wave_f = pop_waveform();
		pair wave = wave_f.pos;
		const int pattern_i = wave_f.state;

		// Check all overlayed tiles.
		for(size_t overlay=0; overlay < overlay_count; overlay++) {
			pair wave_o;

			// If periodic, wrap positions past the edge of the board.
			if (periodic_) {
				wave_o = (wave + overlays[overlay])%wave_shape;
			} else {
				wave_o = wave + overlays[overlay];
			}

			const int wave_o_i = get_idx(wave_o, wave_shape, num_patterns, 0);
			const int wave_o_i_base = get_idx(wave_o, wave_shape, 1, 0);

			// If position is valid and non-collapsed, then propagate changes through
			// this position (wave_o).
			if (wave_o.non_negative() && wave_o < wave_shape && entropy_[wave_o_i_base] > 1 ) {
				auto valid_patterns = fit_table_[pattern_i * overlay_count + overlay];
				for (int pattern_2: valid_patterns)	{
					if(waves_[wave_o_i + pattern_2]) {
						// Get opposite overlay
						const int compat_idx = (wave_o_i + pattern_2)*overlay_count + overlay;
						compatible_neighbors_[compat_idx]--;

						// If there are no valid neighbors left, this state is impossible.
						if (compatible_neighbors_[compat_idx] == 0) 
							ban_waveform(waveform(wave_o, pattern_2));
					}
				}
			}
		}

		iterations += 1;
		if (iterations % 1000 == 0)
			std::cout << "propagation iteration: " << iterations << ", propagation stack size: " << stack_index_ << std::endl;
	}
}

void Model::stack_waveform(waveform& wave) {
	propagate_stack_[stack_index_] = wave;
	stack_index_+=1;
}

waveform Model::pop_waveform() {
	stack_index_-=1;
	return propagate_stack_[stack_index_];
}

void Model::ban_waveform(waveform& wave)
{
	const int waves_idx = get_idx(wave.pos, wave_shape, num_patterns, wave.state);
	const int wave_i = get_idx(wave.pos, wave_shape, 1, 0);

	// Mark this specific waveform as disallowed, and update neighboring patterns
	// to block propagation through this state.
	waves_[waves_idx] = false;
	for (size_t overlay=0; overlay < overlay_count; overlay++) {
		compatible_neighbors_[waves_idx*overlay_count + overlay] = 0;
	}
	stack_waveform(wave);	// Propagate changes through neighboring positions.

	entropy_[wave_i] -= 1;
}

void Model::clear()
{
	for (size_t wave = 0; wave < wave_shape.size; wave++) {
		for (size_t patt = 0; patt < num_patterns; patt++) {
			waves_[wave * num_patterns + patt] = true;
			for (size_t overlay=0; overlay < overlay_count; overlay++) {
				// Reset count of compatible neighbors in the fit table (to all states)
				compatible_neighbors_[(wave*num_patterns + patt)*overlay_count + overlay] = fit_table_[patt*overlay_count + (overlay + 2)%overlay_count].size();
			}
		}
		observed_[wave] = -1;
		entropy_[wave] = num_patterns;
	}
}

void Model::create_waveforms(const bool rotate_patterns) {
	for (const auto& tile : templates) {
		const int height = tile.rows;
		const int width = tile.cols;
		int channels = tile.channels();

		// Add all (D x D) subarrays and (if requested) all it's rotations.
		for (int col = 0; col < width + 1 - dim; col++) {
			for (int row = 0; row < height + 1 - dim; row++) {
				auto pattern = tile(cv::Rect(col, row, dim, dim));
				add_pattern(pattern);
				if (rotate_patterns) {
					cv::Mat cpy;
					cv::rotate(pattern, cpy, cv::ROTATE_90_COUNTERCLOCKWISE);
					add_pattern(cpy);
					cv::Mat cpy1;
					cv::rotate(pattern, cpy1, cv::ROTATE_180);
					add_pattern(cpy1);
					cv::Mat cpy2;
					cv::rotate(pattern, cpy2, cv::ROTATE_90_CLOCKWISE);
					add_pattern(cpy2);
				}
			}
		}
	}
}

void Model::add_pattern(const cv::Mat &pattern) {
	const size_t curr_patt_count = patterns.size();
	for (size_t i = 0; i < curr_patt_count; i++) {
		if (patterns_equal(pattern, patterns[i])) {
			counts_[i] += 1;
			return;
		}
	}
	patterns.push_back(pattern);
	counts_.push_back(1);
}

void Model::generate_fit_table() {
	for (const cv::Mat& center_pattern : patterns) {
		for (pair overlay : overlays) {
			std::vector<int> valid_patterns;
			for (size_t i = 0; i < num_patterns; i++) {
				if (overlay_fit(center_pattern, patterns[i], overlay, dim))
				{
					valid_patterns.push_back(i);
				}
			}
			fit_table_.push_back(valid_patterns);
		}
	}
}
