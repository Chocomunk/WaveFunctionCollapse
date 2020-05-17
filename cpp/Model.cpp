#include "Model.h"
#include "Input.h"
#include "WFCUtil.h"

Model::Model(const std::string tile_dir, Pair &output_shape, const char dim, std::vector<Pair> &overlays, const bool rotate_patterns, const bool periodic, const int iteration_limit) : 
periodic_(periodic), dim(dim), iteration_limit(iteration_limit), overlay_count(overlays.size()), img_shape(output_shape), overlays(overlays) {
	load_tiles(tile_dir, tiles);
	create_waveforms(rotate_patterns);

	srand(time(nullptr));

	num_patterns = patterns.size();
	num_patt_2d = Pair(num_patterns, num_patterns);
	wave_shape = Pair(img_shape.x + 1 - dim, img_shape.y + 1 - dim);

	propagate_stack_ = std::vector<WaveForm>(wave_shape.size * num_patterns);
	entropy_ = std::vector<int>(wave_shape.size);
	waves_ = std::vector<char>(wave_shape.size * num_patterns);
	observed_ = std::vector<int>(wave_shape.size);
	compatible_neighbors_ = std::vector<int>(wave_shape.size * num_patterns * overlay_count);

	out_img = cv::Mat(img_shape.x, img_shape.y, tiles[0].type());

	generate_fit_table();
	std::cout << "Input Tiles: " << tiles.size() << std::endl;
	std::cout << "Patterns: " << num_patterns << std::endl;
	std::cout << "Overlay Count: " << overlay_count << std::endl;
	std::cout << "Wave Shape: " << wave_shape.y << " x " << wave_shape.x << std::endl;
}

void Model::generate_image() {
	std::cout << "Called Generate" << std::endl;
	clear();
	Pair lowest_entropy_idx = Pair(rand_int(wave_shape.x), rand_int(wave_shape.y));
	int iteration = 0;
	while ((iteration_limit < 0 || iteration < iteration_limit) && lowest_entropy_idx.non_negative()) {
		observe_wave(lowest_entropy_idx);
		propagate();
		get_lowest_entropy(lowest_entropy_idx);

		iteration += 1;
		if (iteration % 1000 == 0)
			std::cout << "iteration: " << iteration << std::endl;
	}
	std::cout << "Finished Algorithm" << std::endl;

	for (int row = 0; row < wave_shape.y; row++) {
		for (int col = 0; col < wave_shape.x; col++) {
			render_superpositions(row, col);
		}
	}
	std::cout << "Finished Rendering" << std::endl;
}

cv::Mat& Model::get_image() {
	return out_img;
}

void Model::get_lowest_entropy(Pair &idx) {
	int r = -1; int c = -1;
	int lowest_entropy = -1;
	const size_t waves_count = wave_shape.size;
	for (int wave_idx = 1; wave_idx < waves_count; wave_idx++) {
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
	int num_valid_patterns = 0;
	std::vector<int> valid_patt_idxs;
	for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
		if (waves_[idx_row_col_patt_base + patt_idx]) {
			num_valid_patterns += 1;
			valid_patt_idxs.push_back(patt_idx);
		}
	}
	for (int r = row; r < row + dim; r++) {
		for (int c = col; c < col + dim; c++) {
			BGR& bgr = out_img.ptr<BGR>(r)[c];
			if (num_valid_patterns > 0) {
				bgr = BGR(0, 0, 0);
				for (int patt_idx: valid_patt_idxs)
					bgr += patterns[patt_idx].ptr<BGR>(r - row)[c - col] / num_valid_patterns;
			} else {
				bgr = BGR(204, 51, 255);
			}
		}
	}
}

void Model::observe_wave(Pair &wave) {
	const int idx_row_col_patt_base = get_idx(wave, wave_shape, num_patterns, 0);
	int possible_patterns_sum = 0;
	std::vector<int> possible_indices;
	for (int i = 0; i < num_patterns; i++) {
		if (waves_[idx_row_col_patt_base + i]) {
			possible_patterns_sum += counts_[i];
			possible_indices.push_back(i);
		}
	}

	int rnd = rand_int(possible_patterns_sum)+1;
	int collapsed_index;

	for (collapsed_index = 0; collapsed_index < num_patterns && rnd>0; collapsed_index++) {
		if (waves_[idx_row_col_patt_base + collapsed_index]) {
			rnd -= counts_[collapsed_index];
		}
	}
	collapsed_index -= 1;	// Counter-action against additional increment from for-loop

	for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++)
		if (waves_[idx_row_col_patt_base + patt_idx] != (patt_idx == collapsed_index)) ban_waveform(wave, patt_idx);
	observed_[get_idx(wave, wave_shape, 1, 0)] = collapsed_index;
}

void Model::propagate() {
	int iterations = 0;
	while (stack_index_ > 0) {
		const WaveForm waveform = pop_waveform();
		Pair wave = waveform.wave;
		const int pattern_i = waveform.pattern;

		for(int overlay=0; overlay < overlay_count; overlay++)
		{
			Pair wave_o;
			if (periodic_)
			{
				wave_o = (wave + overlays[overlay])%wave_shape;
			} else
			{
				wave_o = wave + overlays[overlay];
			}

			const int wave_o_i = get_idx(wave_o, wave_shape, num_patterns, 0);
			const int wave_o_i_base = get_idx(wave_o, wave_shape, 1, 0);

			if (wave_o.non_negative() && wave_o < wave_shape && entropy_[wave_o_i_base] > 1 )
			{
				auto valid_patterns = fit_table_[pattern_i * overlay_count + overlay];
				for (int pattern_2: valid_patterns)
				{
					if(waves_[wave_o_i + pattern_2])
					{
						// Get opposite overlay
						const int compat_idx = (wave_o_i + pattern_2)*overlay_count + overlay;
						compatible_neighbors_[compat_idx]--;
						if (compatible_neighbors_[compat_idx] == 0) ban_waveform(wave_o, pattern_2);
					}
				}
			}
		}

		iterations += 1;
		if (iterations % 1000 == 0)
			std::cout << "propagation iteration: " << iterations << ", propagation stack size: " << stack_index_ << std::endl;
	}
}

void Model::stack_waveform(WaveForm& waveform) {
	propagate_stack_[stack_index_] = waveform;
	stack_index_+=1;
}

WaveForm Model::pop_waveform() {
	const WaveForm out = propagate_stack_[stack_index_-1];
	stack_index_-=1;
	return out;
}

void Model::ban_waveform(Pair& wave, const int pattern_i)
{
	const int waves_idx = get_idx(wave, wave_shape, num_patterns, pattern_i);
	const int wave_i = get_idx(wave, wave_shape, 1, 0);
	waves_[waves_idx] = false;
	for (int overlay=0; overlay < overlay_count; overlay++)
	{
		compatible_neighbors_[waves_idx*overlay_count + overlay] = 0;
	}
	stack_waveform(WaveForm(wave, pattern_i));

	entropy_[wave_i] -= 1;
}

void Model::clear()
{
	for (int wave = 0; wave < wave_shape.size; wave++)
	{
		for (int patt = 0; patt < num_patterns; patt++)
		{
			waves_[wave * num_patterns + patt] = true;
			for (int overlay=0; overlay < overlay_count; overlay++)
			{
				// Set count of compatible neighbors to length of valid patterns in the fit table
				compatible_neighbors_[(wave*num_patterns + patt)*overlay_count + overlay] = fit_table_[patt*overlay_count + (overlay + 2)%overlay_count].size();
			}
		}
		observed_[wave] = -1;
		entropy_[wave] = num_patterns;
	}
}

void Model::create_waveforms(const bool rotate_patterns) {
	for (const auto& tile : tiles) {
		const int height = tile.rows;
		const int width = tile.cols;
		int channels = tile.channels();
		for (int col = 0; col < width + 1 - dim; col++) {
			for (int row = 0; row < height + 1 - dim; row++) {
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
	const size_t curr_patt_count = patterns.size();
	for (int i = 0; i < curr_patt_count; i++) {
		if (patterns_equal(waveform, patterns[i])) {
			counts_[i] += 1;
			return;
		}
	}
	patterns.push_back(waveform);
	counts_.push_back(1);
}

void Model::generate_fit_table() {
	for (const cv::Mat& center_pattern : patterns) {
		for (Pair overlay : overlays) {
			std::vector<int> valid_patterns;
			for (int i = 0; i < num_patterns; i++) {
				if (overlay_fit(center_pattern, patterns[i], overlay, dim))
				{
					valid_patterns.push_back(i);
				}
			}
			fit_table_.push_back(valid_patterns);
		}
	}
}
