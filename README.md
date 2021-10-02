# NARv

NARv is a noval CEGAR( counterexample-guided abstraction refinement ) based neural network verification tool.  Inspired by *[An Abstraction-Based Framework for Neural Network Verification](https://doi.org/10.1007/978-3-030-53288-8_3)*, our tool Implemented a new pre-process method and new abstraction-refinement  algorithms.  

Main different from previous work:

- In pre-process, each neuron will be split into two neurons instead of four. 
- We proposed a new abstract operation *freeze*.
- Our algorithm consider the activation range of each neuron as the evaluation of abstract/refine operation.

## Download

```
git clone /repo/outTool.git
```

## Build and Dependencies

NARv is developed based on another CEGAR-based neural network verification tool, [**https://doi.org/10.1007/978-3-030-53288-8_3**] , which use [Marabou](https://github.com/NeuralNetworkVerification/Marabou) python API as network file parser. Our tool inherit it's file parser and data structures, so we need install Marabou first.

#### Marabou python API Installation

To clone Marabou to local, run:

```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
```

To build build both Marabou and Maraboupy using CMake, run:

```
cd path/to/marabou/repo/folder
mkdir build 
cd build
cmake .. -DBUILD_PYTHON=ON
cmake --build .
```

After building, add the Marabou root directory to your PYTHONPATH environmental variable. You can run:

```
export PYTHONPATH=/path/to/marabou/root/
```

## Getting Started

Now, marabou is available as solve engine,  

you can run example by shell in floder **test**:

```
sh test_example.sh
```