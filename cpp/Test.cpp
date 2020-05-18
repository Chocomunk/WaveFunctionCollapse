#include "Input.h"
#include "Model.h"
#include "WFCUtil.h"
#include "Output.h"
#include <opencv2/opencv.hpp>

#define TILES_DIR "../../../tiles/spirals"
#define TILE_DIM 3
#define ROTATE true
#define PERIODIC true
#define WIDTH 64
#define HEIGHT 64
//#define SHOWPATTS

int main() {
	// The set of overlays describing how to compare two patterns. Stored
	// as an (x,y) shift. Shape: [O]
	std::vector<pair> overlays;
	generate_neighbor_overlay(overlays);
	
	// The set of input images to use as templates. Shape: [T]
	std::vector<cv::Mat> template_imgs;
	load_tiles(TILES_DIR, template_imgs);


	// The set of patterns/tiles taken from the input templates. Shape: [N]
	std::vector<cv::Mat> patterns;
	std::vector<int> counts;
	create_waveforms(template_imgs, TILE_DIM, ROTATE, patterns, counts);

	// Stores the set of allowed patterns for a given center pattern and
	// overlay. Stored like an adjacency list. Shape: [N, O][*]
	std::vector<std::vector<int>> fit_table;
	generate_fit_table(patterns, overlays, TILE_DIM, fit_table);
	
	Model model(pair(WIDTH, HEIGHT), 
		patterns.size(), overlays.size(), 
		TILE_DIM, PERIODIC);

#ifdef SHOWPATTS
	bool break_all = false;
	std::cout << std::endl;

	for (size_t pat_idx1 = 0; pat_idx1 < model.num_patterns; pat_idx1++) {
		cv::Mat scaled1;
		auto patt1 = model.patterns[pat_idx1];
		cv::resize(patt1, scaled1, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
		std::cout << "Pattern: " << pat_idx1 << " | Pattern count: " << model.counts[pat_idx1] << std::endl;
		
		for (size_t overlay_idx = 0; overlay_idx < model.overlay_count; overlay_idx++) {
			size_t index = pat_idx1 * model.overlay_count + overlay_idx;
			size_t opposite_idx = pat_idx1 * model.overlay_count + ((overlay_idx + 2) % model.overlay_count);
			auto valid_patterns = model.fit_table[index];
			pair overlay = overlays[overlay_idx];
			pair opposite = overlays[(overlay_idx + 2) % model.overlay_count];
			std::cout << "Valid Patterns: ";
			for (int pattern_2: valid_patterns)	std::cout << pattern_2 << ", ";
			std::cout << std::endl;
			std::cout << "Compat Table Length: " << model.compatible_neighbors_[opposite_idx] << std::endl;
			std::cout << "Overlay: " << overlay << " | Opposite: " << opposite << std::endl;

			bool break_part = false;

			for (size_t pat_idx2 = 0; pat_idx2 < model.num_patterns; pat_idx2++) {
				auto patt2 = model.patterns[pat_idx2];
				cv::Mat scaled2;
				cv::resize(patt2, scaled2, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
				cv::Mat comb;
				cv::hconcat(scaled1, scaled2, comb);
				std::cout << "template: " << pat_idx1 << ", conv: " << pat_idx2 << ", result:" << std::endl;

				std::cout << "    " << "Table: " << (std::find(valid_patterns.begin(), valid_patterns.end(), pat_idx2) != valid_patterns.end()) << " | Calc: " 
					<< overlay_fit(patt1, patt2, overlay, model.dim) << std::endl;

				std::cout << "    " << patterns_equal(patt1, patt2) << ": " << "Patterns are equal" << std::endl;

				cv::imshow("comparison", comb);
				int k = cv::waitKey(0);
				if (k == 27) {
					break_all = true;
					break_part = true;
					break;
				}
				else if(k == int('m'))
				{
					break_part = true;
					break;
				}
				else if (k == int('n')) break;
			}
			std::cout << std::endl;
			if (break_all || break_part) break;
		}
		if (break_all) break;
	}
#endif // SHOWPATTS

	model.generate(overlays, counts, fit_table);

	// Initialize blank output image
	cv::Mat result = cv::Mat(WIDTH, WIDTH, template_imgs[0].type());

	render_image(model, patterns, result);
	
	cv::resize(result, result, cv::Size(800, 800), 0.0, 0.0, cv::INTER_AREA);
	cv::imshow("result", result);
	cv::waitKey(0);

	std::ostringstream outputDir;
	outputDir << TILES_DIR << "/results/cpp/result.png";
	cv::imwrite(outputDir.str(), result);
	
	return 0;
}
