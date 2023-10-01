NARv receives a `.nnet` or `onnx` format neural network as input, it is parsed by the methods in the `nnet/` folder.
The parsed network is represented as the network structure in the `data_structures/` folder, it is also called a 
"full network" (without abstraction). 

Files in the `data_structures/` folder:

+ File `ARNode.py` stands for "abstraction-refinement node", represents a node in the "abstract network", and includes
the key `split_inc_dec` function.
+ File `Layer.py` represents a layer of nodes, and includes key "abstract" and "split" functions.
+ File `Network.py` represents the network structure, and manipulation.

Files in the `pre_process` folder are used before abstraction for splitting nodes into `inc` and/or `dec` nodes.

Files in the `abstraction` folder `merge` or `freeze` nodes.

Files in the `refinement` folder refine (split back) abstract nodes.

Folder `utils` contains helper functions, testing properties, and Marabou API utilities.
