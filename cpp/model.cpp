#include "model.h"

namespace wfc
{
	Model::Model(Pair &output_shape, const int num_patterns, const int overlay_count, 
			const char dim, const bool periodic, const int iteration_limit) :
	dim(dim), iteration_limit(iteration_limit), num_patterns(num_patterns),
	overlay_count(overlay_count), wave_shape(output_shape.x + 1 - dim, output_shape.y + 1 - dim),
	num_patt_2d(num_patterns, num_patterns), periodic_(periodic) {
		srand(time(nullptr));

		// Initialize collections.
		propagate_stack_ = std::vector<Waveform>(wave_shape.size * num_patterns);
		entropy_ = std::vector<int>(wave_shape.size);
		waves_ = std::vector<char>(wave_shape.size * num_patterns);
		observed_ = std::vector<int>(wave_shape.size);
		compatible_neighbors_ = std::vector<int>(wave_shape.size * num_patterns * overlay_count);

		std::cout << "Patterns: " << num_patterns << std::endl;
		std::cout << "Overlay Count: " << overlay_count << std::endl;
		std::cout << "Wave Shape: " << wave_shape.y << " x " << wave_shape.x << std::endl;
	}

	void Model::generate(std::vector<Pair> &overlays, std::vector<int> &counts, std::vector<std::vector<int>> &fit_table) {
		std::cout << "Called Generate" << std::endl;

		// Initialize board into complete superposition, and pick a random wave to collapse
		clear(fit_table);
		Pair lowest_entropy_idx = Pair(rand_int(wave_shape.x), rand_int(wave_shape.y));
		int iteration = 0;
		while ((iteration_limit < 0 || iteration < iteration_limit) && lowest_entropy_idx.non_negative()) {
			/* Standard wfc Loop:
			 *		1. Observe a wave and collapse it's state
			 *		2. Propagate the changes throughout the board and update superposition
			 *		   to only allowed states
			 *		3. After the board state has stabilized, find the position of
			 *		   lowest entropy (most likely to be observed) for the next
			 *		   observation.
			 */
			observe_wave(lowest_entropy_idx, counts);
			propagate(overlays, fit_table);
			get_lowest_entropy(lowest_entropy_idx);

			iteration += 1;
			if (iteration % 1000 == 0)
				std::cout << "iteration: " << iteration << std::endl;
		}
		std::cout << "Finished Algorithm" << std::endl;
	}

	void Model::get_superposition(const int row, const int col, std::vector<int> &patt_idxs) {
		const int idx_row_col_patt_base = row*wave_shape.x*num_patterns + col*num_patterns;

		// Determines the superposition of patterns at this position.
		int num_valid_patterns = 0;
		for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
			if (waves_[patt_idx + idx_row_col_patt_base]) {
				num_valid_patterns += 1;
				patt_idxs.push_back(patt_idx);
			}
		}
	}

	void Model::clear(std::vector<std::vector<int>> &fit_table) {
		for (int wave = 0; wave < wave_shape.size; wave++) {
			for (int patt = 0; patt < num_patterns; patt++) {
				waves_[wave * num_patterns + patt] = true;
				for (int overlay=0; overlay < overlay_count; overlay++) {
					// Reset count of compatible neighbors in the fit table (to all states)
					compatible_neighbors_[(wave*num_patterns + patt)*overlay_count + overlay] = fit_table[patt*overlay_count + (overlay + 2)%overlay_count].size();
				}
			}
			observed_[wave] = -1;
			entropy_[wave] = num_patterns;
		}
	}

	void Model::get_lowest_entropy(Pair &idx) {
		int r = -1; int c = -1;
		int lowest_entropy = -1;
		const int waves_count = wave_shape.size;

		// Checks all non-collapsed positions to find the position of lowest entropy.
		for (int wave_idx = 1; wave_idx < waves_count; wave_idx++) {
			const int entropy_val = entropy_[wave_idx];
			if ((lowest_entropy < 0 || entropy_val < lowest_entropy) && entropy_val > 0 && observed_[wave_idx] == -1) {
				lowest_entropy = entropy_val;
				r = wave_idx/wave_shape.x; c = wave_idx%wave_shape.x;
			}
		}
		idx.x = c;; idx.y = r;
	}

	void Model::observe_wave(Pair &pos, std::vector<int> &counts) {
		const int idx_row_col_patt_base = get_idx(pos, wave_shape, num_patterns, 0);

		// Determines superposition of states and their total frequency counts.
		int possible_patterns_sum = 0;
		std::vector<int> possible_indices;
		for (int i = 0; i < num_patterns; i++) {
			if (waves_[idx_row_col_patt_base + i]) {
				possible_patterns_sum += counts[i];
				possible_indices.push_back(i);
			}
		}

		int rnd = rand_int(possible_patterns_sum)+1;
		int collapsed_index;

		// Randomly selects a state for collapse. Weighted by state frequency count.
		for (collapsed_index = 0; collapsed_index < num_patterns && rnd>0; collapsed_index++) {
			if (waves_[collapsed_index + idx_row_col_patt_base]) {
				rnd -= counts[collapsed_index];
			}
		}
		collapsed_index -= 1;	// Counter-action against additional increment from for-loop

		// Bans all other states, since we have collapsed to a single state.
		for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
			if (waves_[patt_idx + idx_row_col_patt_base] != (patt_idx == collapsed_index)) 
				ban_waveform(Waveform(pos, patt_idx));
		}

		// Assigns the final state of this position.
		observed_[get_idx(pos, wave_shape, 1, 0)] = collapsed_index;
	}

	void Model::propagate(std::vector<Pair>& overlays, std::vector<std::vector<int>> &fit_table) {
		int iterations = 0;
		while (stack_index_ > 0) {
			const Waveform wave_f = pop_waveform();
			Pair wave = wave_f.pos;
			const int pattern_i = wave_f.state;

			// Check all overlayed tiles.
			for(int overlay=0; overlay < overlay_count; overlay++) {
				Pair wave_o;

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
					auto valid_patterns = fit_table[pattern_i * overlay_count + overlay];
					for (int pattern_2: valid_patterns)	{
						if(waves_[wave_o_i + pattern_2]) {
							// Get opposite overlay
							const int compat_idx = (wave_o_i + pattern_2)*overlay_count + overlay;
							compatible_neighbors_[compat_idx]--;

							// If there are no valid neighbors left, this state is impossible.
							if (compatible_neighbors_[compat_idx] == 0) 
								ban_waveform(Waveform(wave_o, pattern_2));
						}
					}
				}
			}

			iterations += 1;
			if (iterations % 1000 == 0)
				std::cout << "propagation iteration: " << iterations << ", propagation stack size: " << stack_index_ << std::endl;
		}
	}

	void Model::stack_waveform(Waveform& wave) {
		propagate_stack_[stack_index_] = wave;
		stack_index_+=1;
	}

	Waveform Model::pop_waveform() {
		stack_index_-=1;
		return propagate_stack_[stack_index_];
	}

	void Model::ban_waveform(Waveform wave) {
		const int waves_idx = get_idx(wave.pos, wave_shape, num_patterns, wave.state);
		const int wave_i = get_idx(wave.pos, wave_shape, 1, 0);

		// Mark this specific Waveform as disallowed, and update neighboring patterns
		// to block propagation through this state.
		waves_[waves_idx] = false;
		for (int overlay=0; overlay < overlay_count; overlay++) {
			compatible_neighbors_[waves_idx*overlay_count + overlay] = 0;
		}
		stack_waveform(wave);	// Propagate changes through neighboring positions.

		entropy_[wave_i] -= 1;
	}
}
