#pragma once
#include <memory>

typedef struct
{
    unsigned int rows, cols;
    std::unique_ptr<unsigned int> data_host;
    unsigned int *data;
    unsigned int elem_size = 1;
    bool is_allocated = false;
} Matrix;

void writeMatrix(const Matrix &mat, const std::string &output)
{
    std::ofstream output_file(output);
    if (!output_file.good())
    {
        std::cerr << "Error: cannot open file " << output << std::endl;
        return;
    }

    output_file << mat.rows << " " << mat.cols << std::endl;
    for (unsigned int i = 0; i < mat.rows; i++)
    {
        for (unsigned int j = 0; j < mat.cols; j++)
        {
            output_file << mat.data_host.get()[i * mat.cols + j] << " ";
        }
        output_file << std::endl;
    }
    output_file.close();
}