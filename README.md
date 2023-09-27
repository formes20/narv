# NARv

**NARv**, short for _**N**etwork **A**bstraction-**R**efinement for **v**erification_, is a neural network verification tool based on structure-oriented CEGAR (counterexample-guided abstraction refinement). The technique details can be found in [[1]](#1) and it is inspired by [[2]](#2).

## Dependencies

#### [Marabou](https://github.com/NeuralNetworkVerification/Marabou) (checked compatibility with commit [30b23b9](https://github.com/NeuralNetworkVerification/Marabou/tree/30b23b9dd59c7656e61f3cf8b04d8ba4996d0cbb))

- Build both Marabou and Maraboupy:
```
cd /path/to/marabou/folder
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON
cmake --build .
```

- Export maraboupy folder to Python path: 
```
PYTHONPATH=PYTHONPATH:/path/to/marabou/folder
```

## Getting Started

You can try an example in the [`tests`](tests/) folder:
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


## References
1. <a id="1"></a> 
Jiaxiang Liu, Yunhan Xing, Xiaomu Shi, Fu Song, Zhiwu Xu, Zhong Ming:
Abstraction and Refinement: Towards Scalable and Exact Verification of Neural Networks. [CoRR abs/2207.00759](https://doi.org/10.48550/arXiv.2207.00759) (2022)

2. <a id="2"></a> Yizhak Yisrael Elboher, Justin Gottschlich, Guy Katz:
An Abstraction-Based Framework for Neural Network Verification. CAV (1) 2020: 43-65
