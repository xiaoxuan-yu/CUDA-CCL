/*
 * © 2024 Xiaoxuan Yu. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 * Main function to execute the algorithms and measure the time taken
 * Foundations of Parallel Computing II, Spring 2024.
 */
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include "Matrix.cuh"
#include "KE.cu"
#include "timer.cxx"
#include "alg_runner.cuh"
#define BLOCK_COLS 16
#define BLOCK_ROWS 16

void init_problem(Matrix &A, Matrix &L, unsigned int rows, unsigned int cols)
{
    A.rows = rows;
    A.cols = cols;
    A.data_host = std::unique_ptr<unsigned int>(new unsigned int[rows * cols]);
    L.rows = rows;
    L.cols = cols;
    L.data_host = std::unique_ptr<unsigned int>(new unsigned int[rows * cols]);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << "<input> <output>\n";
        return EXIT_FAILURE;
    }

    const char *input = argv[1];
    const char *output = argv[2];

    // output file name to string
    std::string output_str;
    output_str = output;

    Matrix A, L;
    unsigned int rows, cols;

    std::fstream input_file;
    input_file.open(input, std::ios::in);
    if (!input_file.good())
    {
        std::cerr << "Error: cannot open file " << input << std::endl;
        return EXIT_FAILURE;
    }

    input_file >> rows >> cols;

    // init matrix
    init_problem(A, L, rows, cols);

    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            input_file >> A.data_host.get()[i * cols + j];
        }
    }

    input_file.close();

    // grid and block size
    dim3 grid_size_KE = dim3((A.cols + BLOCK_COLS - 1) / BLOCK_COLS, (A.rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
    dim3 block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);

    // Execute the algorithms once for validation and CUDA warm up
    // KE algorithm
    Timer KE_timer = KE(A, L, grid_size_KE, block_size_, 0);
    std::string KE_output = "KE_" + output_str;
    writeMatrix(L, KE_output);
    std::cout << "KE time: " << KE_timer.duration << " us" << std::endl;

    // Performance test
    // KE performance test
    Timer KE_performance_timer = performance_test(A, L, "KE", grid_size_KE, block_size_);
    std::cout << "KE performance test time: " << KE_performance_timer.duration << " us" << std::endl;
}