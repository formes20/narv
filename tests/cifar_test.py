#/usr/bin/python3

"""
run one experiment - query cegarabou engine:
calculate if property (p1 or p2) is sat/unsat in a net which is represented by a given .nnet formatted file
write to result file the result of the query and data on the calculation process - times, sizes, etc. .

usage: python3 -f <nnet_filename> -a <abstraction_type> -r <test_refinement type> -e <epsilon value> -l <lower_bound> -o? -p?
example of usage: python3 -f ACASXU_run2a_1_8_batch_2000.nnet -a heuristic -r cegar -e 1e-5 -l 25000 -o -p -s 100
"""

# external imports
def unpickle(file, j):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        input = {}
        for i in range(3072):
            input[i] = dict[b'data'][j][i]/255
        label = dict[b'labels'][j]
    print(input, label)
    return input, label

unpickle("./oval/cifar-10-python/cifar-10-batches-py/data_batch_1",2)