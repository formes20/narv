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

you can run example:

```
python3 one_experiment_global.py -nn 1_1 -pid adversarial_0 -m marabou_with_ar -a global -r global 
```

-nn : network 

-pid：property id

-m： **marabou** or **marabou_with_ar**    with and without abstract-refine mechanism

-a :     abstract algorithm  (only needed by **marabou_with_ar**)

-r : 	refine algorithm    (only needed by **marabou_with_ar**)

details of parameters can be found in one_experiment_global.py

### NARv specification format:

Specifications are written as objects in root/core/utils/verification_properties_utils.py

example:

```python
    "basic_4": {
        "type": "basic",
        "input":
            [
                (0, {"Lower": 0.6, "Upper": 0.6798577687}),
                (1, {"Lower": -0.5, "Upper": 0.5}),
                (2, {"Lower": -0.5, "Upper": 0.5}),
                (3, {"Lower": 0.45, "Upper": 0.5}),
                (4, {"Lower": -0.5, "Upper": -0.45}),
            ],
        "output":
            [
                (0, {"Lower": 1000.0}),
            ]
    },
```

It specify range of inputs and 

type includes ["basic", "adversarial"]

**basic** type specify range of inputs and outputs, 

**adversarial** type specify input range, and the object label (with minimum lower bound) which the classifier should return.

### Run acasxu benchmarks:

```
python3 one_experiment_global.py -nn NET_NAME -pid PROPERTY_NAME -m marabou_with_ar -a global -r global 
```

NET_NAME options : [1_1, 1_2, ..., 1_9, 2_1,..., 2_9, 3_1 ... 5_9]

### Run mnist/cifar benchmarks:

```
python3 one_experiment_global.py -nn cifar400 -pid PROPERTY_NAME -m marabou_with_ar -a global -r global
```


