#pragma once
#include "Matrix.cuh"
#include "UFTree.cuh"
#include "timer.cxx"
#include <iostream>

// Initialize

__global__ void Init_KE(const unsigned int *img, unsigned int *labels, const int cols, const int rows)
{
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_index = row * cols + col;
    unsigned labels_index = row * cols + col;

    if (row < rows && col < cols)
    {
        if (row > 0 && img[img_index - cols] == img[img_index])
        {
            labels[labels_index] = labels_index - cols + 1;
        }

        else if (row > 0 && col > 0 && img[img_index - cols - 1] == img[img_index])
        {
            labels[labels_index] = labels_index - cols;
        }

        else if (row > 0 && col < cols - 1 && img[img_index - cols + 1] == img[img_index])
        {
            labels[labels_index] = labels_index - cols + 2;
        }

        else if (col > 0 && img[img_index - 1] == img[img_index])
        {
            labels[labels_index] = labels_index;
        }

        else
        {
            labels[labels_index] = labels_index + 1;
        }
    }
}

__global__ void Compression_KE(unsigned int *labels, const int cols, const int rows)
{
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned labels_index = row * cols + col;

    if (row < rows && col < cols)
    {
        if (unsigned int label = labels[labels_index])
        {
            labels[labels_index] = Find_label(labels, labels_index, label) + 1;
        }
    }
}

__global__ void Reduce_KE(unsigned int *img, unsigned int *labels, const int cols, const int rows)
{
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_index = row * cols + col;
    unsigned labels_index = row * cols + col;

    if (row < rows && col < cols)
    {
        if (col > 0 && img[img_index - 1] == img[img_index])
        {
            Union(labels, labels_index, labels_index - 1);
        }
        if (row > 0 && col < cols - 1 && img[img_index - cols + 1] == img[img_index])
        {
            Union(labels, labels_index, labels_index - cols + 1);
        }
    }
}

Timer KE(Matrix &A, Matrix &L, dim3 grid_size, dim3 block_size, int i)
{
    int cols = A.cols;
    int rows = A.rows;
    if (A.is_allocated == false)
    {
        cudaMalloc(&A.data, rows * cols * sizeof(unsigned int));
        A.is_allocated = true;
    }
    if (L.is_allocated == false)
    {
        cudaMalloc(&L.data, rows * cols * sizeof(unsigned int));
        L.is_allocated = true;
    }

    // start timer
    Timer timer;

    cudaMemcpy(A.data, A.data_host.get(), rows * cols * sizeof(unsigned int), cudaMemcpyHostToDevice);
    Init_L<<<grid_size, block_size>>>(A.data, L.data, cols, rows);
    Init_KE<<<grid_size, block_size>>>(A.data, L.data, cols, rows);
    Compression_KE<<<grid_size, block_size>>>(L.data, cols, rows);
    Reduce_KE<<<grid_size, block_size>>>(A.data, L.data, cols, rows);
    Compression_KE<<<grid_size, block_size>>>(L.data, cols, rows);
    cudaMemcpy(L.data_host.get(), L.data, rows * cols * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // stop timer
    timer.stop();
    return timer;
}
