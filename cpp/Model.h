#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "WFCUtil.h"

/* Dimension legend
	Template counts: T
	Pattern counts: N
	Tile/Pattern dim: D
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

	/**
	 * \brief The set of input images to use as templates.
	 * 
	 * Shape: [T]
	 */
	std::vector<cv::Mat> templates;

	/**
	 * \brief The set of patterns/tiles taken from the input templates
	 *
	 * Shape: [N]
	 */
	std::vector<cv::Mat> patterns;
	
	/**
	 * \brief The set of overlays describing how to compare two patterns. Stored
	 * as an (x,y) shift.
	 *
	 * Shape: [O]
	 */
	std::vector<pair> overlays;

	cv::Mat out_img;

private:
	int stack_index_ = 0;
	bool periodic_;

	/**
	 * \brief Fixed-space workspace stack for propagation step.
	 *
	 * Shape: [WX * WY * N]
	 */
	std::vector<waveform> propagate_stack_;

	/**
	 * \brief Store the number of times each pattern occurs in the template image.
	 *
	 * Shape: [N]
	 */
	std::vector<int> counts_;

	/**
	 * \brief Stores the entropy (number of valid patterns) for a given position.
	 *
	 * Shape: [WX, WY]
	 */
	std::vector<int> entropy_;

	/**
	 * \brief Stores the set of allowed patterns for a given center pattern and
	 * overlay. Stored like an adjacency list.
	 *
	 * Shape: [N, O][*]
	 */
	std::vector<std::vector<int>> fit_table_;

	/**
	 * \brief Stores whether a specific waveform (position, state) is allowed
	 * (true/false).
	 *
	 * Shape: [WX, WY, N]
	 */
	std::vector<char> waves_;

	/**
	 * \brief Stores the index of the final collapsed pattern for a given position.
	 *
	 * Shape: [WX, WY]
	 */
	std::vector<int> observed_;

	/**
	 * \brief Stores a count of the number of compatible neighbors for this pattern.
	 * If there are no compatible neighbors, then it is impossible for this pattern
	 * to occur and we should ban it.
	 *
	 * Shape: [WX, WY, N, O]
	 */
	std::vector<int> compatible_neighbors_;

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
	 * \brief Adds all (D x D) tiles in the input image to the internal set of
	 * pattern/states. A (5 x 3) input image has 3 (3 x 3) considered tiles.
	 */
	void create_waveforms(bool rotate_patterns);
	
	/**
	 * \brief Adds the given (D x D) tile, to the internal set of patterns/states.
	 * Duplicates are counted to keep track of the frequencies of unique patterns.
	 */
	void add_pattern(const cv::Mat &pattern);
	
	/**
	 * \brief Given the internal set of patterns/states, generates the overlay
	 * constraints (can be/cannot be overlayed) for every pair of states.
	 */
	void generate_fit_table();
};