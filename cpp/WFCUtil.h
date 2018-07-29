#pragma once
#include <vector>
#include <opencv2\opencv.hpp>

struct Pair {
	int x; int y; int size;

public:
	explicit Pair(int x=0, int y=0);
	friend std::ostream& operator<<(std::ostream& os, const Pair& obj);
	inline Pair operator+(const Pair& other) const;
	inline Pair operator+(const int val) const;
	inline bool operator<(const Pair& other) const;
	inline bool non_negative() const;
	inline bool shifted_less_than(const Pair& other, const int shift) const;
};

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

void generate_sliding_overlay(char dim, std::vector<Pair> &out);
void generate_neighbor_overlay(std::vector<Pair> &out);

bool patterns_equal(const cv::Mat &patt1, const cv::Mat &patt2);
bool overlay_fit(const cv::Mat &patt1, const cv::Mat &patt2, Pair &overlay, char dim);

inline int get_idx(Pair &idx, Pair &dim, int depth, int depth_idx);
int rand_int(int max_val);


inline Pair Pair::operator+(const Pair& other) const
{
	return Pair(this->x + other.x, this->y + other.y);
}

inline Pair Pair::operator+(const int val) const
{
	return Pair(this->x + val, this->y + val);
}

inline bool Pair::operator<(const Pair& other) const
{
	return this->x < other.x && this->y < other.y;
}

inline bool Pair::non_negative() const
{
	return this->x >= 0 && this->y >= 0;
}

inline bool Pair::shifted_less_than(const Pair& other, const int shift) const
{
	return this->x + shift < other.x && this->y + shift < other.y;
}

inline int get_idx(Pair &idx, Pair &dim, int depth, int depth_idx) {
	return idx.y * dim.x * depth + idx.x * depth + depth_idx;
}
