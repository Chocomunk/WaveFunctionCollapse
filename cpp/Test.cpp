#include "Input.h"
#include "Model.h"
#include <opencv2/opencv.hpp>

#define TILES_DIR "../../../tiles/spirals"
//#define SHOWPATTS

int main() {
	std::vector<Pair> overlays;
	generate_neighbor_overlay(overlays);
	Model model(TILES_DIR, Pair(64, 64), 3, overlays, true, true, -1);
	
#ifdef SHOWPATTS
	auto break_all = false;
	std::cout << std::endl;

	for (auto pat_idx1 = 0; pat_idx1 < model.num_patterns; pat_idx1++) {
		cv::Mat scaled1;
		auto patt1 = model.patterns[pat_idx1];
		cv::resize(patt1, scaled1, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
		std::cout << "Pattern: " << pat_idx1 << " | Pattern count: " << model.counts_[pat_idx1] << std::endl;
		
		for (auto overlay_idx = 0; overlay_idx < model.overlay_count; overlay_idx++) {
			int index = pat_idx1 * model.overlay_count + overlay_idx;
			int opposite_idx = pat_idx1 * model.overlay_count + ((overlay_idx + 2) % model.overlay_count);
			auto valid_patterns = model.fit_table_[index];
			auto overlay = overlays[overlay_idx];
			auto opposite = overlays[(overlay_idx + 2) % model.overlay_count];
			std::cout << "Valid Patterns: ";
			for (auto pattern_2: valid_patterns)	std::cout << pattern_2 << ", ";
			std::cout << std::endl;
			std::cout << "Compat Table Length: " << model.compatible_neighbors_[opposite_idx] << std::endl;
			std::cout << "Overlay: " << overlay << " | Opposite: " << opposite << std::endl;

			auto break_part = false;

			for (auto pat_idx2 = 0; pat_idx2 < model.num_patterns; pat_idx2++) {
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
				auto k = cv::waitKey(0);
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

	model.generate_image();
	auto result = model.get_image();
	cv::resize(result, result, cv::Size(800, 800), 0.0, 0.0, cv::INTER_AREA);
	cv::imshow("result", result);
	cv::waitKey(0);

	std::ostringstream outputDir;
	outputDir << TILES_DIR << "/results/cpp/result.png";
	cv::imwrite(outputDir.str(), result);
	
	return 0;
}
