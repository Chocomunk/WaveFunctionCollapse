# Wave Function Collapse: Procedural Tile Generation

This is a C++ and (separate) python port of https://github.com/mxgmn/WaveFunctionCollapse.

All contained template images also come from https://github.com/mxgmn/WaveFunctionCollapse.

## Getting Started
To build the cpp project with g++ on linux, run:

`make build`

This will generate a `bin/wfc` executable. This repository comes with template images (credited above), which can be used to test the compiled executable. We have included 5 test cases, which can all be executed by running:

`make test`

The output images are stored in the `results/` folder.

## Requirements
This project was most recently built with [OpenCV 4.3.0](https://docs.opencv.org/4.3.0/), which is the only dependency. On our systems, we installed OpenCV using the following command:

`sudo apt-get install libopencv-dev`

In the case that the compiler cannot find OpenCV, you may need to change the `$(OPENCV)` variable in the makefile to any of `opencv['',1,2,3,4]`.

## The WFC Algorithm

**Note:** This is just a brief description of the WFC algorithm. For a detailed explanation, see https://github.com/mxgmn/WaveFunctionCollapse, re-writing mxgmn's explanation while giving due credit would be redundant.

This is a procedural generation algorithm. The implementation contained in this repository operates on images; it generates an output that is locally similar to the input. Currently, the algorithm takes a single template image, cuts it up into multiple tiles, and gives it to the WFC algorithm as input. A simple modification for the future is to take existing tiles and neighboring constraints as direct inputs to the program.

The WFC model takes a set of states with neighboring constraints as input. The goal of the algorithm is to generate a board of tiles such that all tiles are collapsed to a single state, and the board satisfies all neighboring constraints.

All tiles in the output are initialized to a superposition of all states in the input. The algorithm procedurally collapses each tile to a single state via the following loop:
1. Perform a measurement (observation) on the tile of lowest entropy to collapse it to a single state.
2. Propagate the changes caused by the collapsed tile throughout the board. This step continues until all current constraints are satisfied, and there are no changes left to propagate.
3. Determine the tile of lowest entropy (least uncertainty on the states) for the next loop iteration.
