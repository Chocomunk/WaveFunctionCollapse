#include <string>
#include <vector>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <stack>
#include "Input.h"
#include "WFCUtil.h"
#include <random>

/* Documentation legend
Tile count: T
Pattern dim: N
Overlay counts: O
Wave shape x: WX
Wave shape y: WY
*/

class Model {

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

	std::stack<Pair> propagate_stack;

	std::vector<int> counts;				// Shape: [N]
	std::vector<int> entropies;				// Shape: [WX, WY]
	std::vector<char> fit_table;			// Shape: [N, N, O]
	std::vector<char> waves;				// Shape: [WX, WY, N]
	std::vector<char> observed;				// Shape: [WX, WY]
	std::vector<char> registered_propagate;	// Shape: [WX, WY]

	cv::Mat out_img;

public:
	Model(std::string tile_dir, Pair &output_shape, char dim, std::vector<Pair> &overlays, bool rotate_patterns=false, int iteration_limit=-1);
	void generate_image();
	cv::Mat& get_image();

public:
	void get_lowest_entropy(Pair &idx);
	void render_superpositions(int row, int col);
	void observe_wave(Pair &wavef);
	void propagate();
	void register_waveform(Pair &waveform);
	Pair pop_waveform();
	void update_wave(Pair &wavef, int overlay_idx, std::vector<int> &valid_indices);

	void create_waveforms(bool rotate_patterns);
	void add_waveform(const cv::Mat &waveform);
	void generate_fit_table();

};