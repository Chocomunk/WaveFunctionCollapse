#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * \brief Represents a pair of integral values.
 */
struct pair {
	size_t x; size_t y; size_t size;

public:
	pair(size_t x=0, size_t y=0);
	
	/**
	 * \return True if neither x nor y are negative, False otherwise
	 */
	inline bool non_negative() const;
	
	/**
	 * \return True if '(this + shift) < other', False otherwise
	 */
	inline bool shifted_less_than(const pair& other, const size_t shift) const;

	friend std::ostream& operator<<(std::ostream& os, const pair& obj);
	inline pair operator+(const pair& other) const;
	inline pair operator+(const size_t val) const;
	inline pair operator%(const pair& other) const;
	inline bool operator<(const pair& other) const;
};

/**
 * \brief Represents a wave state at position 'pair(x,y)' in state 'state'
 */
struct waveform
{
	pair pos; int state;

public:
	waveform(pair pos, int state);
};

/**
 * \brief Represents a pixel color value
 */
struct BGR final
{
	uchar b;
	uchar g;
	uchar r;

public:
	BGR(uchar b, uchar g, uchar r);
	BGR operator/(int val) const;
	friend std::ostream& operator<<(std::ostream& os, const BGR& obj);
	void operator+=(BGR& other);
};

/**
 * \brief Generates an overlay of all possible intersecting overlays of two patterns
 */
void generate_sliding_overlay(char dim, std::vector<pair> &out);

/**
 * \brief Generates an overlay of the patterns shifted one unit to the top, bottom,
 * left, and right.
 */
void generate_neighbor_overlay(std::vector<pair> &out);

/**
 * \return True if both patterns have the same pixel values
 */
bool patterns_equal(const cv::Mat &patt1, const cv::Mat &patt2);

/**
 * \return True if we can lay patt2 on patt1 with the given overlay position
 */
bool overlay_fit(const cv::Mat &patt1, const cv::Mat &patt2, pair &overlay, char dim);

/**
 * \return The effective index of the flattened [dim.x, dim.y, depth] array.
 */
inline int get_idx(pair &idx, pair &dim, int depth, int depth_idx);

/**
 * \return A random integer within [0, max_val)
 */
int rand_int(int max_val);


inline bool pair::non_negative() const
{
	return this->x >= 0 && this->y >= 0;
}

inline bool pair::shifted_less_than(const pair& other, const size_t shift) const
{
	return this->x + shift < other.x && this->y + shift < other.y;
}

inline int get_idx(pair &idx, pair &dim, int depth, int depth_idx) {
	return idx.y * dim.x * depth + idx.x * depth + depth_idx;
}
inline pair pair::operator+(const pair& other) const
{
	return pair(this->x + other.x, this->y + other.y);
}

inline pair pair::operator+(const size_t val) const
{
	return pair(this->x + val, this->y + val);
}

inline pair pair::operator%(const pair& other) const
{
	auto x=this->x, y=this->y;
	if (x >= other.x) x -= other.x;
	if (x < 0) x += other.x;
	if (y >= other.y) y -= other.y;
	if (y < 0) y += other.y;
	return pair(x, y);
}

inline bool pair::operator<(const pair& other) const
{
	return this->x < other.x && this->y < other.y;
}
