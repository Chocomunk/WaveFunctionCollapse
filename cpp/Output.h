#pragma once
#include "Model.h"

/**
 * \brief Renders the board state of the given model into an output image. Patterns
 * must be ordered the same way as it's counts are passed into the model.
 */
void render_image(Model& model, std::vector<cv::Mat>& patterns, cv::Mat& out_img,
	int width, int height, int dim);
