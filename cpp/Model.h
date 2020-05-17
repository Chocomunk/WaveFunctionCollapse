#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "WFCUtil.h"

/* Dimension legend
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
	pair img_shape;
	pair wave_shape;
	pair num_patt_2d;

	std::vector<cv::Mat> tiles;					// Shape: [T]
	std::vector<cv::Mat> patterns;				// Shape: [N]
	std::vector<pair> overlays;					// Shape: [O]

	cv::Mat out_img;

private:
	int stack_index_ = 0;
	bool periodic_;

	std::vector<waveform> propagate_stack_;		// Shape: [WX * WY * N]

	std::vector<int> counts_;					// Shape: [N]
	std::vector<int> entropy_;					// Shape: [WX, WY]
	std::vector<std::vector<int>> fit_table_;	// Shape: [N, O]
	std::vector<char> waves_;					// Shape: [WX, WY, N]
	std::vector<int> observed_;					// Shape: [WX, WY]
	std::vector<int> compatible_neighbors_;		// Shape: [WX, WY, N, O]

public:
	/**
	 * \brief Initializes a model instance and allocates all workspace data.
	 */
	Model(pair &output_shape, std::string tile_dir, char dim, std::vector<pair> &overlays, 
		  bool rotate_patterns=false, bool periodic=false, int iteration_limit=-1);
	
	/**
	 * \brief Runs the WFC algorithm and stores the output image (call 'get_image'
	 * to access).
	 */
	void generate_image();
	
	/**
	 * \brief Resets all tiles to a perfect superposition.
	 */
	void clear();
	
	/**
	 * \return The stored output image (may be blank if 'generate_image' has not
	 * been called).
	 */
	cv::Mat& get_image();

private:
	/**
	 * \brief Finds the wave with lowest entropy and stores it's position in idx
	 */
	void get_lowest_entropy(pair &idx);
	
	/**
	 * \brief Generates an image of the superpositions of the wave at (row, col),
	 * and stores the result in the output image.
	 */
	void render_superpositions(int row, int col);
	
	/**
	 * \brief Performs an observation on the wave at the given position and
	 * collapses it to a single state.
	 */
	void observe_wave(pair &pos);
	
	/**
	 * \brief Iteratively collapses waves in the tilemap until no conflicts exist.
	 * Meant to be used after collapsing a wave by observing it.
	 */
	void propagate();

	/**
	 * \brief Adds a wave to the propagation stack to propagate changes to it's
	 * neighbors (determined by overlays).
	 */
	void stack_waveform(waveform& wave);
	
	/**
	 * \return The last waveform added to the propagation stack.
	 */
	waveform pop_waveform();

	/**
	 * \brief Bans a specific state at the waveform position. Partially collapses
	 * the overall state at that position and reduces it's entropy.
	 */
	void ban_waveform(waveform& wave);

	/**
	 * \brief Adds all (N x N) tiles in the input images. A (5 x 3) input
	 * image has 3 (3 x 3) considered tiles.
	 */
	void create_waveforms(bool rotate_patterns);
	
	/**
	 * \brief Adds the given (N x N) tile, to the internal set of patterns/states.
	 * Duplicates are counted to keep track of the frequencies of unique patterns.
	 */
	void add_pattern(const cv::Mat &pattern);
	
	/**
	 * \brief Given the internal set of patterns/states, generates the overlay
	 * constraints (can be/cannot be overlayed) for every pair of states.
	 */
	void generate_fit_table();
};