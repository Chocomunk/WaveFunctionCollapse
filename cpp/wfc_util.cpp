#include "wfc_util.h"

namespace wfc
{
	Pair::Pair(int x, int y) : x(x), y(y), size(x * y) {}

	std::ostream& operator<<(std::ostream& os, const Pair& obj) {
		os << "(" << obj.x << ", " << obj.y << ")";
		return os;
	}

	Waveform::Waveform(const Pair pos, const int state): pos(pos), state(state){}

	BGR::BGR(const uchar b, const uchar g, const uchar r) : b(b), g(g), r(r) {}

	BGR BGR::operator/(const int val) const
	{
		return BGR(b / val, g / val, r / val);
	}

	std::ostream& operator<<(std::ostream& os, const BGR& obj) {
		os << "<" << static_cast<int>(obj.b) << ", " << static_cast<int>(obj.g) << ", " << static_cast<int>(obj.r) << ">";
		return os;
	}

	void BGR::operator+=(BGR other) {
		this->b += other.b;
		this->g += other.g;
		this->r += other.r;
	}

	void generate_sliding_overlay(const char dim, std::vector<Pair> &out) {
		for (int i = 1 - dim; i < dim; i++) {
			for (int j = 1 - dim; j < dim; j++) {
				if (i != 0 || j != 0) {
					out.emplace_back(i, j);
				}
			}
		}
	}

	void generate_neighbor_overlay(std::vector<Pair> &out) {
		out.clear();
		out.emplace_back(-1,  0);
		out.emplace_back(0,  1);
		out.emplace_back(1,  0);
		out.emplace_back(0, -1);
	}

	void generate_fit_table(const std::vector<cv::Mat> &patterns, const std::vector<Pair> &overlays, 
		const int dim, std::vector<std::vector<int>> &fit_table) {
		size_t num_patterns = patterns.size();
		for (const cv::Mat& center_pattern : patterns) {
			for (Pair overlay : overlays) {
				std::vector<int> valid_patterns;
				for (size_t i = 0; i < num_patterns; i++) {
					if (overlay_fit(center_pattern, patterns[i], overlay, dim))
					{
						valid_patterns.push_back(i);
					}
				}
				fit_table.push_back(valid_patterns);
			}
		}
	}

	bool overlay_fit(const cv::Mat &patt1, const cv::Mat &patt2, Pair &overlay, char dim) {
		CV_Assert(patt1.depth() == patt2.depth() &&
			patt1.depth() == CV_8U &&
			patt1.channels() == patt2.channels() &&
			patt1.cols == patt2.cols && 
			patt1.rows == patt2.rows &&
			patt1.cols == dim &&
			patt1.rows == dim);

		/* 
		 * Computes the range of indices at which to patt1 overlaps with patt2. Indices
		 * are computed from the given overlay and are in terms of patt1 indexing.
		 * To convert to patt2 indexing, subtract by the row/col shifts (shown blow).
		 */
		
		const int channels = patt1.channels();
		int row_shift = overlay.y, col_shift = overlay.x;

		const int row_start = MAX(row_shift, 0);
		const int row_end = MIN(row_shift + dim - 1, dim - 1) + 1;
		int col_start = MAX(col_shift, 0);
		int col_end = MIN(col_shift + dim - 1, dim - 1) + 1;

		col_start *= channels;
		col_end *= channels;
		col_shift *= channels;

		// Row equivalent for patt2 is {value} - row_shift
		// Col equivalent for patt2 is {value} - col_shift

		// Checks if the overlapping pixels are equal.
		int i, j;
		const uchar* p1; const uchar* p2;
		bool matching = true;
		for (i = row_start; i < row_end && matching; i++) {
			p1 = patt1.ptr<uchar>(i);
			p2 = patt2.ptr<uchar>(i - row_shift);
			for (j = col_start; j < col_end && matching; j++) {
				matching = p1[j] == p2[j-col_shift];
			}
		}

		return matching;
	}

	int rand_int(const int max_val) {
		return rand() % max_val;
	}
}
