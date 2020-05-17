#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "WFCUtil.h"

/* Documentation legend
Tile count: T
Pattern dim: N
Overlay counts: O
Wave shape x: WX
Wave shape y: WY
*/

class Model {

private:
	int stack_index_ = 0;
	bool periodic_;

	std::vector<WaveForm> propagate_stack_;	// Shape: [WX * WY * N]

	std::vector<int> counts_;				// Shape: [N]
	std::vector<int> entropy_;			// Shape: [WX, WY]
	std::vector<std::vector<int>> fit_table_;	// Shape: [N, O][*]
	std::vector<char> waves_;				// Shape: [WX, WY, N]
	std::vector<int> observed_;			// Shape: [WX, WY]
	std::vector<int> compatible_neighbors_; // Shape: [WX, WY, N, O]

public:
	const char dim;
	const int iteration_limit;
	size_t num_patterns;
	size_t overlay_count;
	Pair img_shape;
	Pair wave_shape;
	Pair num_patt_2d;

	std::vector<cv::Mat> tiles;				// Shape: [T]
	std::vector<cv::Mat> patterns;			// Shape: [N]
	std::vector<Pair> overlays;				// Shape: [O]

	cv::Mat out_img;

public:
	Model(std::string tile_dir, Pair &output_shape, char dim, std::vector<Pair> &overlays, bool rotate_patterns=false, bool periodic=false, int iteration_limit=-1);
	void generate_image();
	void clear();
	cv::Mat& get_image();

private:
	void get_lowest_entropy(Pair &idx);
	void render_superpositions(int row, int col);
	void observe_wave(Pair &wave);
	void propagate();
	void stack_waveform(WaveForm& waveform);
	WaveForm pop_waveform();
	void ban_waveform(Pair& wave, int pattern_i);

	void create_waveforms(bool rotate_patterns);
	void add_waveform(const cv::Mat &waveform);
	void generate_fit_table();

};