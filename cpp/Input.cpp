#include "input.h"

void load_tiles(std::string dirname, std::vector<cv::Mat> &out){
	std::vector<cv::String> filenames;
	cv::glob(dirname + "/*.png", filenames, false);
	size_t count = filenames.size();
	for (size_t i = 0; i < count; i++) {
		out.push_back(cv::imread(filenames[i]));
	}
}

void create_waveforms(const std::vector<cv::Mat> &templates, const int dim, 
	const bool rotate_patterns, 
	std::vector<cv::Mat> &patterns, std::vector<int> &counts) {
	for (const auto& tile : templates) {
		const int height = tile.rows;
		const int width = tile.cols;
		int channels = tile.channels();

		// Add all (D x D) subarrays and (if requested) all it's rotations.
		for (int col = 0; col < width + 1 - dim; col++) {
			for (int row = 0; row < height + 1 - dim; row++) {
				auto pattern = tile(cv::Rect(col, row, dim, dim));
				add_pattern(pattern, patterns, counts);
				if (rotate_patterns) {
					cv::Mat cpy;
					cv::rotate(pattern, cpy, cv::ROTATE_90_COUNTERCLOCKWISE);
					add_pattern(cpy, patterns, counts);
					cv::Mat cpy1;
					cv::rotate(pattern, cpy1, cv::ROTATE_180);
					add_pattern(cpy1, patterns, counts);
					cv::Mat cpy2;
					cv::rotate(pattern, cpy2, cv::ROTATE_90_CLOCKWISE);
					add_pattern(cpy2, patterns, counts);
				}
			}
		}
	}
}

void add_pattern(const cv::Mat &pattern, std::vector<cv::Mat> &patterns, std::vector<int> &counts) {
	const size_t curr_patt_count = patterns.size();
	for (size_t i = 0; i < curr_patt_count; i++) {
		if (patterns_equal(pattern, patterns[i])) {
			counts[i] += 1;
			return;
		}
	}
	patterns.push_back(pattern);
	counts.push_back(1);
}

bool patterns_equal(const cv::Mat &patt1, const cv::Mat &patt2) {
	CV_Assert(patt1.depth() == patt2.depth() &&
		patt1.depth() == CV_8U &&
		patt1.channels() == patt2.channels() &&
		patt1.cols == patt2.cols && patt1.rows == patt2.rows);
	const int channels = patt1.channels();
	int n_rows = patt1.rows;
	int n_cols = patt1.cols * channels;

	if (patt1.isContinuous() && patt2.isContinuous()) {
		n_cols *= n_rows;
		n_rows = 1;
	}

	// Checks each pixel value for equality
	int i, j;
	const uchar* p1; const uchar* p2;
	bool matching = true;
	for (i = 0; i < n_rows && matching; i++) {	
		p1 = patt1.ptr<uchar>(i);
		p2 = patt2.ptr<uchar>(i);
		for (j = 0; j < n_cols && matching; j++) {
			matching = p1[j] == p2[j];
		}
	}

	return matching;
}