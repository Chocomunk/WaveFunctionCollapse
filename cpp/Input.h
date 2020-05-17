#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * \brief Stores all png images in a directory to a vector.
 */
void load_tiles(std::string dirname, std::vector<cv::Mat> &out);