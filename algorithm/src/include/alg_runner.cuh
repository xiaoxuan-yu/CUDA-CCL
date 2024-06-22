/*
 * execute the connected components algorithm and measure the time taken， the algorithm is provided via function pointer
 * Xiaoxuan, Yu, parcoii2024
 */

#pragma once
#include "timer.cxx"
#include "../alg/KE.cu"
#include "Matrix.cuh"
#include <iostream>

#define num_replications 100

inline Timer performance_test(Matrix &A, Matrix &L, const std::string &algo, dim3 grid_size, dim3 block_size)
{
    Timer timer;
    if (algo == "KE")
    {
        for (int i = 0; i < num_replications; i++)
        {
            timer = timer + KE(A, L, grid_size, block_size, i);
        }
    }
    else
    {
        std::cerr << "Error: unknown algorithm " << algo << std::endl;
        exit(EXIT_FAILURE);
    }
    return timer / num_replications;
}