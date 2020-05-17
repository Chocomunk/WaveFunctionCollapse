#include "WFCUtil.h"

pair::pair(size_t x, size_t y) : x(x), y(y), size(x * y) {}

std::ostream& operator<<(std::ostream& os, const pair& obj) {
	os << "(" << obj.x << ", " << obj.y << ")";
	return os;
}

waveform::waveform(const pair pos, const int state): pos(pos), state(state){}

BGR::BGR(const uchar b, const uchar g, const uchar r) : b(b), g(g), r(r) {}

BGR BGR::operator/(const int val) const
{
	return BGR(b / val, g / val, r / val);
}

std::ostream& operator<<(std::ostream& os, const BGR& obj) {
	os << "<" << static_cast<int>(obj.b) << ", " << static_cast<int>(obj.g) << ", " << static_cast<int>(obj.r) << ">";
	return os;
}

void BGR::operator+=(BGR& other) {
	this->b += other.b;
	this->g += other.g;
	this->r += other.r;
}

void generate_sliding_overlay(const char dim, std::vector<pair> &out) {
	for (int i = 1 - dim; i < dim; i++) {
		for (int j = 1 - dim; j < dim; j++) {
			if (i != 0 || j != 0) {
				out.emplace_back(i, j);
			}
		}
	}
}

void generate_neighbor_overlay(std::vector<pair> &out) {
	out.clear();
	out.emplace_back(-1,  0);
	out.emplace_back(0,  1);
	out.emplace_back(1,  0);
	out.emplace_back(0, -1);
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

bool overlay_fit(const cv::Mat &patt1, const cv::Mat &patt2, pair &overlay, char dim) {
	CV_Assert(patt1.depth() == patt2.depth() &&
		patt1.depth() == CV_8U &&
		patt1.channels() == patt2.channels() &&
		patt1.cols == patt2.cols && 
		patt1.rows == patt2.rows &&
		patt1.cols == dim &&
		patt1.rows == dim);

	const int channels = patt1.channels();
	int row_shift = overlay.y, col_shift = overlay.x;

	const int row_start_1 = MAX(row_shift, 0);
	const int row_end_1 = MIN(row_shift + dim - 1, dim - 1) + 1;
	int col_start_1 = MAX(col_shift, 0);
	int col_end_1 = MIN(col_shift + dim - 1, dim - 1) + 1;

	col_start_1 *= channels;
	col_end_1 *= channels;
	col_shift *= channels;

	// Row equivalent for patt2 is {value} - row_shift
	// Col equivalent for patt2 is {value} - col_shift

	int i, j;
	const uchar* p1; const uchar* p2;
	bool matching = true;
	for (i = row_start_1; i < row_end_1 && matching; i++) {
		p1 = patt1.ptr<uchar>(i);
		p2 = patt2.ptr<uchar>(i - row_shift);
		for (j = col_start_1; j < col_end_1 && matching; j++) {
			matching = p1[j] == p2[j-col_shift];
		}
	}

	return matching;
}

int rand_int(const int max_val) {
	return rand() % max_val;
}