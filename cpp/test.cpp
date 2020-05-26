#include "input.h"
#include "wfc.h"
#include <opencv2/opencv.hpp>

using namespace wfc;

int main(int argc, char** argv) {
	char* tiles_dir;
	int tile_dim = 3;
	int rotate = 1;
	int periodic = 1;
	int width = 64;
	int height = 64;
	int render = 0;

	if (!(argc > 1)) {
		std::cout << "Usage {arg_name (options) | default}:" << std::endl <<
			"\twfc {image folder} {tile dim | 3} {rotate? (0/1) | 1} {periodic? (0/1) | 1} {width | 64} {height | 64} {render? (0/1) | 0}"
			<< std::endl;
		return -1;
	}

	tiles_dir = argv[1];
	if (argc > 2)
		tile_dim = atoi(argv[2]); // denotes tile dimension
	if (argc > 3)
		rotate = atoi(argv[3]); // 0 for no rotation, 1 for rotation
	if (argc > 4)
		periodic = atoi(argv[4]); // 0 if not periodic, 1 for periodic 
	if (argc > 5)
		width = atoi(argv[5]); // denotes tile width
	if (argc > 6)
		height = atoi(argv[6]); // denotes tile height;
	if (argc > 7)
		render = atoi(argv[7]); // render toggle

	// The set of overlays describing how to compare two patterns. Stored
	// as an (x,y) shift. Shape: [O]
	std::vector<Pair> overlays;
	generate_neighbor_overlay(overlays);

	// The set of input images to use as templates. Shape: [T]
	std::vector<cv::Mat> template_imgs;
	load_tiles(tiles_dir, template_imgs);


	// The set of patterns/tiles taken from the input templates. Shape: [N]
	std::vector<cv::Mat> patterns;
	std::vector<int> counts;
	create_waveforms(template_imgs, tile_dim, rotate, patterns, counts);

	// Stores the set of allowed patterns for a given center pattern and
	// overlay. Stored like an adjacency list. Shape: [N, O][*]
	std::vector<std::vector<int>> fit_table;
	generate_fit_table(patterns, overlays, tile_dim, fit_table);

	Pair p = Pair(width, height);
	Model model(p,
	            patterns.size(), overlays.size(),
	            tile_dim, periodic);

	// Shows all patterns
	if (render) {
		std::cout << "Pattern Viewer Controls:" << std::endl <<
			"\tEsc: Quit and continue" << std::endl <<
			"\tM: Next left pattern" << std::endl <<
			"\t(Any Other Key): Next right pattern" << std::endl;

		bool break_all = false;
		std::cout << std::endl;

		for (int pat_idx1 = 0; pat_idx1 < model.num_patterns; pat_idx1++) {
			cv::Mat scaled1;
			auto patt1 = patterns[pat_idx1];
			cv::resize(patt1, scaled1, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
			std::cout << "Pattern: " << pat_idx1 << " | Pattern count: " << patterns[pat_idx1] << std::endl;

			for (int overlay_idx = 0; overlay_idx < model.overlay_count; overlay_idx++) {
				size_t index = pat_idx1 * model.overlay_count + overlay_idx;
				auto valid_patterns = fit_table[index];
				Pair overlay = overlays[overlay_idx];
				Pair opposite = overlays[(overlay_idx + 2) % model.overlay_count];
				std::cout << "Valid Patterns: ";
				for (int pattern_2 : valid_patterns) std::cout << pattern_2 << ", ";
				std::cout << std::endl;
				std::cout << "Overlay: " << overlay << " | Opposite: " << opposite << std::endl;

				bool break_part = false;

				for (int pat_idx2 = 0; pat_idx2 < model.num_patterns; pat_idx2++) {
					auto patt2 = patterns[pat_idx2];
					cv::Mat scaled2;
					cv::resize(patt2, scaled2, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
					cv::Mat comb;
					cv::hconcat(scaled1, scaled2, comb);
					std::cout << "template: " << pat_idx1 << ", conv: " << pat_idx2 << ", result:" << std::endl;

					std::cout << "    " << "Table: " << (std::find(valid_patterns.begin(), valid_patterns.end(),
					                                               pat_idx2) != valid_patterns.end()) << " | Calc: "
						<< overlay_fit(patt1, patt2, overlay, model.dim) << std::endl;

					std::cout << "    " << patterns_equal(patt1, patt2) << ": " << "Patterns are equal" << std::endl;

					cv::imshow("comparison", comb);
					int k = cv::waitKey(0);
					if (k == 27) {
						break_all = true;
						break_part = true;
						break;
					}
					else if (k == int('m')) {
						break_part = true;
						break;
					}
					// else if (k == int('n')) break;
				}
				std::cout << std::endl;
				if (break_all || break_part) break;
			}
			if (break_all) break;
		}
	}

	model.generate(overlays, counts, fit_table);

	// Initialize blank output image
	cv::Mat result = cv::Mat(width, width, template_imgs[0].type());

	render_image(model, patterns, result);

	cv::resize(result, result, cv::Size(800, 800), 0.0, 0.0, cv::INTER_AREA);
	cv::imshow("result", result);
	cv::waitKey(0);

	std::ostringstream outputDir;
	outputDir << tiles_dir << "/results/cpp/result.png";
	cv::imwrite(outputDir.str(), result);

	return 0;
}
