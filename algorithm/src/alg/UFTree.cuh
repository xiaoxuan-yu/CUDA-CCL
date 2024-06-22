/*
 * © 2024 Xiaoxuan Yu. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 * UFTree data structure and related functions
 * Foundations of Parallel Computing II, Spring 2024.
 */
#pragma once
// find the root index of the UF tree
__device__ unsigned Find(const unsigned int *buf, unsigned n)
{
    unsigned label = buf[n];

    while (label - 1 != n)
    {
        n = label - 1;
        label = buf[n];
    }
    return n;
}

// return the root index of the UF tree
__device__ unsigned Find_label(const unsigned int *buf, unsigned n, unsigned label)
{

    while (label - 1 != n)
    {
        n = label - 1;
        label = buf[n];
    }
    return n;
}

// Merge the UFTrees
__device__ void Union(unsigned int *buf, unsigned n1, unsigned n2)
{
    bool complete;
    do
    {
        n1 = Find(buf, n1);
        n2 = Find(buf, n2);

        if (n1 < n2)
        {
            int old = atomicMin(buf + n2, n1 + 1);
            complete = (old == n2 + 1);
            n2 = old - 1;
        }
        else if (n1 > n2)
        {
            int old = atomicMin(buf + n1, n2 + 1);
            complete = (old == n1 + 1);
            n1 = old - 1;
        }
        else
        {
            complete = true;
        }
    } while (!complete);
}

__global__ void Init_L(const unsigned int *img, unsigned int *labels, const int cols, const int rows)
{
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int labels_index = row * cols + col;

    if (row < rows && col < cols)
    {
        labels[labels_index] = 0;
    }
}