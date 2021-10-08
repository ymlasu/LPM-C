# LPM-C
An implementation of a nonlocal lattice particle method (LPM) using an iterative solution procedure

## Prerequisites
1. C++ software development environment
2. Intel MKL library

## Instructions

### Install OneAPI
1. source /your-oneapi-directory/setvars.sh linux64 --force
2. install Ninja

### Compile and run LPM-C
1. Clone the https://github.com/longfish/LPM-C.git into your own machine (prefer linux)
2. mkdir build & cd build
3. cmake .. -G "Ninja" -DMKL_INTERFACE=lp64
4. cmake --build . -j8

### Run the code
./lpmc

### Check the result
The result are in the parent folder.

### Examples
In the examples folder, check the different loading cases, change the name into 'lpmc_project.c' and replace it with the same file in src directory.

### References
Meng C, Wei H, Chen H, et al. Modeling plasticity of cubic crystals using a nonlocal lattice particle method[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 385: 114069.

Meng C, Liu Y. Nonlocal Damage-enhanced Plasticity Model for Ductile Fracture Analysis Using a Lattice Particle Method[J]. arXiv preprint arXiv:2108.01214, 2021.