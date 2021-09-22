import numpy as np
def read_cifar():
    with open("cifar6", 'w') as file_object:
        for i in [2,4,6,8,10,12,14]:
            weight= np.load("./"+str(i)+"_weight.npy").tolist()
            bias = np.load("./"+str(i)+"_bias.npy").tolist()
            print(len(weight))
            print(type(bias))
            for node in range(len(weight)):
                for edge in range(len(weight[node])):
                    file_object.write(str(weight[node][edge]))
                    file_object.write(",")
                file_object.write("\n")
            for bias_index in range(len(bias)):
                file_object.write(str(bias[bias_index]))
                file_object.write(",")
                file_object.write("\n")
            # for layer_index in range(network_length):
            #     if layer_index == 0:
            #         for node in network.layers[layer_index].nodes:
            #             file_object.write("Input {}".format(node.name))
            #             file_object.write("\n")
            #     elif layer_index == network_length-1:
            #         for node in network.layers[layer_index].nodes:
            #             file_object.write("Linear {} ".format(node.name))
            #             file_object.write("{} ".format(node.bias))
            #             for edge in node.in_edges:
            #                 file_object.write("{} {} ".format(edge.weight, edge.src))
            #             file_object.write("\n")
            #     else:
            #         for node in network.layers[layer_index].nodes:
            #             file_object.write("ReLU {} ".format(node.name))
            #             file_object.write("{} ".format(node.bias))
            #             for edge in node.in_edges:
            #                 file_object.write("{} {} ".format(edge.weight, edge.src))
            #             file_object.write("\n")
            # for index, bound in property["input"]:
            #     file_object.write("Assert >= {} 1.0 x_0_{}".format(bound["Upper"],index))
            #     file_object.write("\n")
            #     file_object.write("Assert <= {} 1.0 x_0_{}".format(bound["Lower"],index))
            #     file_object.write("\n")
            # for index, bound in property["output"]:
            #     #file_object.write("Assert <= {} 1.0 x_{}_{}".format(network_length-1, bound["upper"],index))
            #     file_object.write("Assert <= {} 1.0 {}".format(bound["Lower"], network.layers[-1].nodes[index].name))
            #     file_object.write("\n")

read_cifar()