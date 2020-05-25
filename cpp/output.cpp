#include "output.h"

namespace wfc
{
	void render_image(Model& model, std::vector<cv::Mat>& patterns, cv::Mat &out_img) {
		const int height = model.wave_shape.y;
		const int width = model.wave_shape.x;
		const int dim = model.dim;
		std::vector<int> valid_patts = std::vector<int>(model.num_patterns);
		
		for (int row=0; row < height; row++) {
			for (int col=0; col < width; col++) {
				// Get the valid patterns for this position and render the
				// average of each pattern
				valid_patts.clear();
				model.get_superposition(row, col, valid_patts);
				for (int r = row; r < row + dim; r++) {
					for (int c = col; c < col + dim; c++) {
						BGR& bgr = out_img.ptr<BGR>(r)[c];
						if (valid_patts.empty()) {
							// Error: No valid patterns (magenta). Generating a board that is
							// guaranteed to have no errors is NP.
							bgr = BGR(204, 51, 255);
						} else {
							bgr = BGR(0, 0, 0);
							for (int patt_idx: valid_patts)
								bgr += patterns[patt_idx].ptr<BGR>(r - row)[c - col] / valid_patts.size();
						}
					}
				}
			}
		}

		std::cout << "Finished Rendering" << std::endl;
	}
}
