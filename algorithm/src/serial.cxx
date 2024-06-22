/*
 * Foundations of Parallel Computing II, Spring 2024.
 * Instructor: Chao Yang, Xiuhong Li @ Peking University.
 * This is a serial implementation of Connected Components Labeling.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "timer.cxx"

#define num_replications 20

typedef struct {
    int** p;        // Each pixel's ancestor node
    int** image;    // Image data
    int rows;       // Number of rows in the image
    int cols;       // Number of columns in the image
} UnionFind;

// Initialize the Union Find structure
void init(UnionFind* uf) {
    for (int i = 0; i < uf->rows; i++) {
        for (int j = 0; j < uf->cols; j++) {
            uf->p[i][j] = i * uf->cols + j;  // Set each pixel to its own set
        }
    }
}

// Find the ancestor of x with path compression
int find(UnionFind* uf, int x) {
    if (uf->p[x / uf->cols][x % uf->cols] != x) {
        uf->p[x / uf->cols][x % uf->cols] = find(uf, uf->p[x / uf->cols][x % uf->cols]);
    }
    return uf->p[x / uf->cols][x % uf->cols];
}

// Merge the sets of two pixels
void union_sets(UnionFind* uf, int x, int y) {
    int rootX = find(uf, x);
    int rootY = find(uf, y);
    if (rootX != rootY) {
        uf->p[rootY / uf->cols][rootY % uf->cols] = rootX;
    }
}

// Connected Components Labeling algorithm
void connected_components_labeling(UnionFind* uf) {
    init(uf);

    for (int i = 0; i < uf->rows; i++) {
        for (int j = 0; j < uf->cols; j++) {
            // Check the surrounding neighbors (8 / 2 = 4 points)
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy < 1; dy++) {
                    if (dx >= 0 && dy == 0) continue; // We only need to check 4 directions
                    int ni = i + dx, nj = j + dy;
                    if (ni >= 0 && ni < uf->rows && nj >= 0 && nj < uf->cols && uf->image[ni][nj] == uf->image[i][j]) {
                        union_sets(uf, i * uf->cols + j, ni * uf->cols + nj);
                    }
                }
            }
        }
    }

    for (int i = 0; i < uf->rows; i++) {
        for (int j = 0; j < uf->cols; j++) {
            find(uf, i * uf->cols + j);
        }
    }
}

// Write the result of the Union Find to a file
void writeResultToFile(UnionFind* uf, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // First line: print the number of rows and columns
    fprintf(fp, "%d %d\n", uf->rows, uf->cols);

    // Starting from the second line, print the result of the Union Find
    for (int i = 0; i < uf->rows; i++) {
        for (int j = 0; j < uf->cols; j++) {
            fprintf(fp, "%d ", uf->p[i][j]);
        }
    }

    fclose(fp);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    UnionFind uf;
    int rows, cols;
    FILE* input_file = fopen(input_filename, "r");
    if (input_file == NULL) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    fscanf(input_file, "%d %d", &rows, &cols);

    uf.p = (int**)malloc(rows * sizeof(int*));
    uf.image = (int**)malloc(rows * sizeof(int*));
    uf.rows = rows;
    uf.cols = cols;

    for (int i = 0; i < rows; i++) {
        uf.p[i] = (int*)malloc(cols * sizeof(int));
        uf.image[i] = (int*)malloc(cols * sizeof(int));
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(input_file, "%d", &uf.image[i][j]);
        }
    }

    fclose(input_file);

    connected_components_labeling(&uf);
    writeResultToFile(&uf, output_filename);

    Timer timer;
    timer.pause();

    for (int i =0; i < num_replications; i++) {
        timer.resume();
        connected_components_labeling(&uf);
        timer.pause();
    }
    
    timer = timer / num_replications;

    std::cout << "Serial performance test time: " << timer.duration << " us" << std::endl;

    for (int i = 0; i < rows; i++) {
        free(uf.p[i]);
        free(uf.image[i]);
    }
    free(uf.p);
    free(uf.image);

    return 0;
}