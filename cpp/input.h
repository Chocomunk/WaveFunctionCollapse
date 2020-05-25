#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * \brief Stores all png images in a directory to a vector.
 */
void load_tiles(std::string dirname, std::vector<cv::Mat> &out);

/**
 * \brief Adds all (D x D) tiles in the input image to the internal set of
 * pattern/states. A (5 x 3) input image has 3 (3 x 3) considered tiles.
 */
void create_waveforms(const std::vector<cv::Mat> &templates, const int dim, 
	const bool rotate_patterns, 
	std::vector<cv::Mat> &patterns, std::vector<int> &counts);

/**
 * \brief Adds the given (D x D) tile, to the internal set of patterns/states.
 * Duplicates are counted to keep track of the frequencies of unique patterns.
 */
void add_pattern(const cv::Mat &pattern, std::vector<cv::Mat> &patterns, std::vector<int> &counts);

/**
 * \return True if both patterns have the same pixel values
 */
bool patterns_equal(const cv::Mat &patt1, const cv::Mat &patt2);