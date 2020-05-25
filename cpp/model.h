#pragma once
#include <vector>
#include "wfc_util.h"

/* Dimension legend
	Template counts: T
	Pattern counts: N
	Tile/Pattern dim: D
	Overlay counts: O
	Wave shape x: WX
	Wave shape y: WY
*/
namespace wfc
{
	class Model {

	public:
		const char dim;
		const int iteration_limit;
		size_t num_patterns;
		size_t overlay_count;
		Pair wave_shape;
		Pair num_patt_2d;

	private:
		int stack_index_ = 0;
		bool periodic_;

		/**
		 * \brief Fixed-space workspace stack for propagation step.
		 *
		 * Shape: [WX * WY * N]
		 */
		std::vector<Waveform> propagate_stack_;

		/**
		 * \brief Stores the entropy (number of valid patterns) for a given position.
		 *
		 * Shape: [WX, WY]
		 */
		std::vector<int> entropy_;

		/**
		 * \brief Stores whether a specific Waveform (position, state) is allowed
		 * (true/false).
		 *
		 * Shape: [WX, WY, N]
		 */
		std::vector<char> waves_;

		/**
		 * \brief Stores the index of the final collapsed pattern for a given position.
		 *
		 * Shape: [WX, WY]
		 */
		std::vector<int> observed_;

		/**
		 * \brief Stores a count of the number of compatible neighbors for this pattern.
		 * If there are no compatible neighbors, then it is impossible for this pattern
		 * to occur and we should ban it.
		 *
		 * Shape: [WX, WY, N, O]
		 */
		std::vector<int> compatible_neighbors_;

	public:
		/**
		 * \brief Initializes a model instance and allocates all workspace data.
		 */
		Model(Pair &output_shape, const int num_patterns, const int overlay_count, 
			const char dim, const bool periodic=false, const int iteration_limit=-1);
		
		/**
		 * \brief Runs the wfc algorithm and stores the output image (call 'get_image'
		 * to access).
		 */
		void generate(std::vector<Pair> &overlays, std::vector<int> &counts, std::vector<std::vector<int>> &fit_table);
		
		/**
		 * \brief Generates an image of the superpositions of the wave at (row, col),
		 * and stores the result in the output image.
		 */
		void get_superposition(int row, int col, std::vector<int> &patt_idxs);
		
		/**
		 * \brief Resets all tiles to a perfect superposition.
		 */
		void clear(std::vector<std::vector<int>> &fit_table);
		
	private:
		/**
		 * \brief Finds the wave with lowest entropy and stores it's position in idx
		 */
		void get_lowest_entropy(Pair &idx);
		
		/**
		 * \brief Performs an observation on the wave at the given position and
		 * collapses it to a single state.
		 */
		void observe_wave(Pair &pos, std::vector<int> &counts);
		
		/**
		 * \brief Iteratively collapses waves in the tilemap until no conflicts exist.
		 * Meant to be used after collapsing a wave by observing it.
		 */
		void propagate(std::vector<Pair>& overlays, std::vector<std::vector<int>> &fit_table);

		/**
		 * \brief Adds a wave to the propagation stack to propagate changes to it's
		 * neighbors (determined by overlays).
		 */
		void stack_waveform(Waveform& wave);
		
		/**
		 * \return The last Waveform added to the propagation stack.
		 */
		Waveform pop_waveform();

		/**
		 * \brief Bans a specific state at the Waveform position. Partially collapses
		 * the overall state at that position and reduces it's entropy.
		 */
		void ban_waveform(Waveform& wave);
	};
}
