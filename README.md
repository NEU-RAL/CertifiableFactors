# Simplifying Certifiable Estimation: A Factor Graph Optimization Approach
by [Zhenxin (Jason) Xu](https://zhexin1904.github.io/), Nikolas R. Sanderson, [David M. Rosen](https://coe.northeastern.edu/people/rosen-david/). We are affiliated with the [Robust Autonomy Lab](https://neural.lab.northeastern.edu/) at Northeastern University.

Factor graphs are the dominant paradigm for
modeling state estimation tasks in mobile robotics, as they
afford both a convenient modular modeling language and fast,
scalable inference algorithms. However, most state-of-the-art
factor graph inference approaches rely on local optimization,
which makes them susceptible to converging to incorrect estimates. 
Recent work has led to the design of novel certifiable optimization 
algorithms capable of efficiently recovering verifiably
globally optimal estimates in practice. However, Despite these
advantages, the widespread adoption of certifiable estimation
methods has been limited by the extensive manual effort
required to custom-design appropriate relaxations and efficient
optimization algorithms. To address these challenges, in this
paper we present a method that leverages the same factor
graph and local optimization framework widely used in robotics
and computer vision to design and deploy a broad range of
certifiable estimators. We describe how to implement lifted
versions of the variable and factor types typically encountered
in robotic mapping and localization problems. The result is a
set of certifiable factors that enables practitioners to develop
and deploy globally optimal estimators with the same ease as
conventional local methods. Experimental results validate our
approach, demonstrating global optimality comparable to that
achieved by state-of-the-art certifiable solvers.

Presented at ICRA 2025 Atlanta Georgia in workshop ["Robots in the Wild"](https://dartmouthrobotics.github.io/icra-2025-robots-wild/).

Full journal article will be submitted later this year. 


## Install

The C++ implementation can be built and exported as a CMake project.

#### C++ quick installation guide

The following installation instructions have been verified on Ubuntu 22.04:

*Step 1:*  Install dependencies

```
sudo apt-get install liblapack-dev libblas-dev libsuitesparse-dev
```
[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)  (Make sure using the same version o external as GTSAM's to avoid conflicts, i.e., you can enable `-DGTSAM_USE_SYSTEM_EIGEN=ON` when complie GTSAM)

[GTSAM](https://github.com/borglab/gtsam) (tested on release/[4.3a0](https://github.com/borglab/gtsam/releases/tag/4.3a0), should also compatible with a series of versions that after **Boost Removal**, we use OptionalMatrixType but not boost optional matrix for customer factors)

*Step 2:*  Clone the repository

```
git clone git@github.com:NEU-RAL/CertifiableFactors.git
```

*Step 3:*  Initialize Git submodules
```
cd CertifiableFactors
git submodule init
git submodule update
```

*Step 4:*  Create build directory

```
mkdir build
```

*Step 5:*  Configure build and generate Makefiles

```
cd build && cmake ..
```

*Step 6:*  Build code

```
make -j
```

*Step 7:*  Run the examples

For example, when running a 2D example(d=2), and set the initial rank as p=d=2. Note that for any examples, p should be >=d.

```
./your_executable d p input_file.g2o output_file.csv
```