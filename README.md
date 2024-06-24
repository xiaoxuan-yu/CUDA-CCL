# CUDA-CCL
> Code and report for the final project of the course *"Foundations of Parallel Computing II"* at Peking University, 2024 Spring.

This repository contains the implementation of 8-connected Komura Equivalence algorithm for Connected Component Labeling (CCL) on CUDA. The algorithm is originally described in

S.  Allegretti,  F.  Bolelli,  M.  Cancilla,  and  C.  Grana,  “Optimizing  GPU­Based  ConnectedComponents Labeling Algorithms,” in 2018 IEEE International Conference on Image Processing,Applications and Systems (IPAS),  2018, pp. 175–180. doi: 10.1109/IPAS.2018.8708900.

## Contained Contents
- `algorithm/`: The implementation of the algorithm.
- `validation`: The python scripts for validating the correctness of the algorithm.
- `perf`: Original data for performance evaluation.
- `report`: `typst` source code and the compiled PDF report.

## Usage
### Prerequisites
- A CUDA compatible GPU
- CUDA Toolkit 11.0 or later (Tested on CUDA 11.3 and CUDA 12.4)
- Supported GCC/G++ required by the CUDA Toolkit (GCC 9 tested for CUDA 11 and GCC 12 tested for CUDA 12)
- CMake 3.19 or later
- (Optional) `Python`, `numpy`, `ipykernel` and `OpenCV` for validation
### Build
1. clone the repository
```bash
git clone https://github.com/xiaoxuan-yu/CUDA-CCL
```
2. `cd` into the repository
```bash
cd CUDA-CCL
```
3. Configure the project with CMake
```bash
cmake -S algorithm -B build
```
4. Build the project
```bash
cmake --build build
```
### Usage
The executable will be generated in `build/bin` directory. The usage is as follows:
```bash
$ ./CCL <input_file> <output_file>
```
### Validation
Use the `validation.ipynb` to validate the correctness of the algorithm.

## Licence
This project is licensed under BSD 3-Clause Licence. See [LICENCE](LICENCE) for more details.

Parto of the code referenced [YACCLAB](https://github.com/prittt/YACCLAB), which is also licensed under BSD 3-Clause Licence.