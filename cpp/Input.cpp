#include "Input.h"

void load_tiles(std::string dirname, std::vector<cv::Mat> &out){
	std::vector<cv::String> filenames;
	cv::glob(dirname + "/*.png", filenames, false);
	size_t count = filenames.size();
	for (size_t i = 0; i < count; i++) {
		out.push_back(cv::imread(filenames[i]));
	}
}