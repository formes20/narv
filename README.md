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

You can try an example in the [`tests/`](tests/) folder:
```
sh test_example.sh
```


## File Tree

```
|--core/ 
|--experiments/
|--planet/
|--tests/
|--narv.py
|--README.md
```

- Folder [`core/`](core/): the main code of NARv.
- Folder [`experiments/`](experiments/): the benchmarks and the experimental data for the TOSEM submission.
- Folder [`planet/`](planet/): the code of the tool [Planet](https://github.com/progirep/planet) from its repository.
- Folder [`tests/`](tests/): some testing scripts for NARv.



## References
1. <a id="1"></a> 
Jiaxiang Liu, Yunhan Xing, Xiaomu Shi, Fu Song, Zhiwu Xu, Zhong Ming:
Abstraction and Refinement: Towards Scalable and Exact Verification of Neural Networks. [CoRR abs/2207.00759](https://doi.org/10.48550/arXiv.2207.00759) (2022)

2. <a id="2"></a> Yizhak Yisrael Elboher, Justin Gottschlich, Guy Katz:
An Abstraction-Based Framework for Neural Network Verification. CAV (1) 2020: 43-65
