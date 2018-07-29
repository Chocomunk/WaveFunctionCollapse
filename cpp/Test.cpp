#include "Input.h"
#include "Model.h"
#include <opencv2/opencv.hpp>

#define TILES_DIR "../../../tiles/flowers"
//#define SHOWPATTS

int main() {
	std::vector<Pair> overlays;
	generate_neighbor_overlay(overlays);
	Model model(TILES_DIR, Pair(128, 128), 3, overlays, false, -1);
	
#ifdef SHOWPATTS
	bool breakAll = false;
	std::cout << std::endl;

	for (int patIdx1 = 0; patIdx1 < model.numPatterns; patIdx1++) {
		cv::Mat scaled1;
		cv::Mat patt1 = model.patterns[patIdx1];j
		cv::resize(patt1, scaled1, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
		std::cout << "Pattern count: " << model.counts[patIdx1] << std::endl;
		for (int patIdx2 = 0; patIdx2 < model.numPatterns; patIdx2++) {
			Pair lookupIdx(patIdx2, patIdx1);
			cv::Mat patt2 = model.patterns[patIdx2];
			cv::Mat scaled2;
			cv::resize(patt2, scaled2, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
			cv::Mat comb;
			cv::hconcat(scaled1, scaled2, comb);
			std::cout << "template: " << patIdx1 << ", conv: " << patIdx2 << ", result:" << std::endl;
			for (int overlayIdx = 0; overlayIdx < model.overlayCount; overlayIdx++) {
				Pair ovly = overlays[overlayIdx];
				int idx = getIdx(lookupIdx, model.numPatt2D, model.overlayCount, overlayIdx);
				std::cout << "    " << idx << " : " << (bool)model.fitTable[idx] << " | " << overlayFit(patt1, patt2, ovly, model.dim) <<  ": " << ovly << std::endl;
			}
			std::cout << "    " << patternsEqual(patt1, patt2) << ": " << "Patterns are equal" << std::endl << std::endl;

			cv::imshow("comparison", comb);
			int k = cv::waitKey(0);
			if (k == 27) {
				breakAll = true;
				break;
			} else if (k == int('n')) {
				break;
			}
		}
		if (breakAll) {
			break;
		}
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
