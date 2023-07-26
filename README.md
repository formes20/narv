# NARv

NARv is a noval CEGAR (counterexample-guided abstraction refinement) based neural network verification tool. 
Inspired by *[An Abstraction-Based Framework for Neural Network Verification](https://doi.org/10.1007/978-3-030-53288-8_3)*, 
NARv implements a new pre-process method and new abstraction-refinement algorithms.  

NARv has the following differences from the above CEGAR-based verification tool:

+ In pre-process, NARv splits each neuron into two neurons instead of four;
+ NARv includes a new abstract operation `freeze`;
+ NARv consider the activation range of each neuron as the evaluation of abstract/refine operation.

## Dependencies

NARv is based on the [Marabou](https://github.com/NeuralNetworkVerification/Marabou) solver,
using its data structures and python API for SAT solving.

This project includes the dependent Marabou in the `Marabou` folder. 
To build both Marabou and Maraboupy, run:

```
cd path/to/marabou/repo/folder
mkdir build 
cd build
cmake .. -DBUILD_PYTHON=ON
cmake --build .
```

After building, export maraboupy folder to Python path: 

```
PYTHONPATH=PYTHONPATH:/path/to/marabou/folder
```

## Getting Started

Now, marabou is available as solve engine, you can run an example in the `tests` folder:

```
sh test_example.sh
```

## Experiments

Folder `experiments` contains the files from all experiments, including inputs (properties), networks, and results.
Each dataset folder has a `narv-*` file, which used to run the corresponding experiment.

## Code Structure

The core pre-process and abstraction-refinement procedure of NARv is placed in the `core` folder:
```
|--core/
|  |--abstraction/
|  |--data_structures/
|  |--nnet/
|  |--pre_process/
|  |--refinement/
|  |--utils/
...
|  |--import_marabou.py
|--narv.py (key func: one_experiment())
```

NARv receives a `.nnet` or `onnx` format neural network as input, it is parsed by the methods in the `core/nnet` folder.
The parsed network is represented as the network structure in the `core/data_structures` folder, it is also called a 
"full network" (without abstraction). 

Files in the `data_structures` folder :

+ File `ARNode.py` stands for "abstraction-refinement node", represents a node in the "abstract network", and includes
the key `split_inc_dec` function.
+ File `Layer.py` represents a layer of nodes, and includes key "abstract" and "split" functions.
+ File `Network.py` represents the network structure, and manipulations.

Files in the `pre_porcess` folder are used before abstraction for splitting nodes into `inc` and / or `dec` nodes.

Files in the `abstraction` folder `merge` or `freeze` nodes.

Files in the `refinement` folder refine (split back) abstract nodes.

Folder `utils` contains helper functions, test properties, and marabou API utilities.

More implementation details of each file can be found in the `doc` folder.
