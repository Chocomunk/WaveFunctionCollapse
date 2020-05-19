#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace wfc
{
	/**
	 * \brief Represents a Pair of integral values.
	 */
	struct Pair {
		int x; int y; int size;

	public:
		Pair(int x=0, int y=0);
		
		/**
		 * \return True if neither x nor y are negative, False otherwise
		 */
		inline bool non_negative() const;
		
		/**
		 * \return True if '(this + shift) < other', False otherwise
		 */
		inline bool shifted_less_than(const Pair& other, const size_t shift) const;

		friend std::ostream& operator<<(std::ostream& os, const Pair& obj);
		inline Pair operator+(const Pair& other) const;
		inline Pair operator+(const int val) const;
		inline Pair operator%(const Pair& other) const;
		inline bool operator<(const Pair& other) const;
	};

	/**
	 * \brief Represents a wave state at position 'Pair(x,y)' in state 'state'
	 */
	struct Waveform
	{
		Pair pos; int state;

	public:
		Waveform(Pair pos = { 0,0 }, int state = 0);
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
	void generate_sliding_overlay(char dim, std::vector<Pair> &out);

	/**
	 * \brief Generates an overlay of the patterns shifted one unit to the top, bottom,
	 * left, and right.
	 */
	void generate_neighbor_overlay(std::vector<Pair> &out);

	/**
	 * \return The effective index of the flattened [dim.x, dim.y, depth] array.
	 */
	inline int get_idx(Pair &idx, Pair &dim, int depth, int depth_idx);
		
	/**
	 * \brief Given the internal set of patterns/states, generates the overlay
	 * constraints (can be/cannot be overlayed) for every Pair of states.
	 */
	void generate_fit_table(const std::vector<cv::Mat> &patterns, const std::vector<Pair> &overlays, 
		const int dim, std::vector<std::vector<int>> &fit_table);

	/**
	 * \return True if we can lay patt2 on patt1 with the given overlay position
	 */
	bool overlay_fit(const cv::Mat &patt1, const cv::Mat &patt2, Pair &overlay, char dim);

	/**
	 * \return A random integer within [0, max_val)
	 */
	int rand_int(int max_val);

	inline bool Pair::non_negative() const
	{
		return this->x >= 0 && this->y >= 0;
	}

	inline bool Pair::shifted_less_than(const Pair& other, const size_t shift) const
	{
		return this->x + shift < other.x && this->y + shift < other.y;
	}

	inline int get_idx(Pair &idx, Pair &dim, int depth, int depth_idx) {
		return idx.y * dim.x * depth + idx.x * depth + depth_idx;
	}
	inline Pair Pair::operator+(const Pair& other) const
	{
		return Pair(this->x + other.x, this->y + other.y);
	}

	inline Pair Pair::operator+(const int val) const
	{
		return Pair(this->x + val, this->y + val);
	}

	inline Pair Pair::operator%(const Pair& other) const
	{
		auto x=this->x, y=this->y;
		if (x >= other.x) x -= other.x;
		if (x < 0) x += other.x;
		if (y >= other.y) y -= other.y;
		if (y < 0) y += other.y;
		return Pair(x, y);
	}

	inline bool Pair::operator<(const Pair& other) const
	{
		return this->x < other.x && this->y < other.y;
	}
}
