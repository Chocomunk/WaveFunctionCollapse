#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

void load_tiles(std::string dirname, std::vector<cv::Mat> &out);