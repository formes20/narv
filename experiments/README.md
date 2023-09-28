This folder contains the benchmarks and the experimental data for the TOSEM submission (ID: TOSEM-2023-0071) entitled "**Abstraction and Refinement: Towards Scalable and Exact Verification of Neural Networks**".


### Folder [`ACASXu`](ACASXu/)

ACAS Xu contains 45 DNNs and is used for Performance Evaluation comparing NARv with CEGAR-NN.

- [**networks/**](ACASXu/networks/) contains the 45 DNNs in the nnet format.
- [**inputs/**](ACASXu/inputs/) contains 4 files representing the inputs for 4 different
  perturbation thresholds ranging from 0.01 to 0.04.  Each file
  includes 20 inputs for that corresponding perturbation threshold.
- [**results/**](ACASXu/results/) contains experimental results for Figures 7 and 8 and
  Table 2 in the submission, organized w.r.t. the perturbation
  thresholds.

  Each csv file named by the tool (e.g. NARv.csv) contains the
  experimental data for that tool. The columns are as follows.

  - *network_id*:  network ID
  - *property_id*:  input ID
  - *abstract_time*:  time (in seconds) for building the initial abstraction
  - *total_time*: verification time (in seconds)
  - *size*: total size of the (abstract) network when the
    verification succeeds


### Folder [`MNIST`](MNIST/)

The DNN trained on MNIST dataset is used for Effectiveness Evaluation
comparing NARv[M] (resp., NARv[P]) with Marabou (resp., Planet).

- [**networks/**](MNIST/networks/) contains the DNN trained on MNIST dataset in the nnet
  format.
- [**inputs/**](MNIST/inputs/) contains 25 inputs. They are used for each perturbation
  threshold, which ranges from 0.02 to 0.05.
- [**results/**](MNIST/results/) contains experimental results for Figure 4 and Table 1
  in the submission, organized w.r.t. the perturbation thresholds.

  Each csv file named by the two tools (e.g. NARv[M]_vs_Marabou.csv)
  contains the experimental data for the comparison between those
  two tools. The columns are as follows.

  - *property_id*:  input ID used for robustness verification
  - *{toolname}_time*: verification time (in seconds) spent by the
    tool {toolname}

### Folder [`CIFAR-10`](CIFAR-10/)

The two DNNs trained on CIFAR-10 dataset, with hidden neurons 4 x 100
and 6 x 100 respectively, are used for Effectiveness Evaluation
comparing NARv[M] (resp., NARv[P]) with Marabou (resp., Planet).

- [**networks/**](CIFAR-10/networks/) contains the two DNN trained on CIFAR-10 dataset in
  the nnet format.
- [**inputs/**](CIFAR-10/inputs/) contains 48 inputs. There are 12 inputs for each
  perturbation threshold, which ranges from 0.001 to 0.004.
- [**results/**](CIFAR-10/results/) contains experimental results for Figures 5 and 6, and 
  Table 1 in the submission, organized w.r.t. the networks
  (4x100 and 6x100, respectively) and then the perturbation
  thresholds.

  Each csv file named by the two tools (e.g. NARv[M]_vs_Marabou.csv)
  contains the experimental data for the comparison between those
  two tools. The columns are as follows.

  - *property_id*:  input ID used for robustness verification
  - *{toolname}_time*: verification time (in seconds) spent by the
    tool {toolname}

