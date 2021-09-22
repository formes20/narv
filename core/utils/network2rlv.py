
def network2rlv(network, property, filename):
    with open(filename, 'w') as file_object:
        network_length = len(network.layers)
        for layer_index in range(network_length):
            if layer_index == 0:
                for node in network.layers[layer_index].nodes:
                    file_object.write("Input {}".format(node.name))
                    file_object.write("\n")
            elif layer_index == network_length-1:
                for node in network.layers[layer_index].nodes:
                    file_object.write("Linear {} ".format(node.name))
                    file_object.write("{} ".format(node.bias))
                    for edge in node.in_edges:
                        file_object.write("{} {} ".format(edge.weight, edge.src))
                    file_object.write("\n")
            else:
                for node in network.layers[layer_index].nodes:
                    file_object.write("ReLU {} ".format(node.name))
                    file_object.write("{} ".format(node.bias))
                    for edge in node.in_edges:
                        file_object.write("{} {} ".format(edge.weight, edge.src))
                    file_object.write("\n")
        for index, bound in property["input"]:
            file_object.write("Assert >= {} 1.0 x_0_{}".format(bound["Upper"],index))
            file_object.write("\n")
            file_object.write("Assert <= {} 1.0 x_0_{}".format(bound["Lower"],index))
            file_object.write("\n")
        for index, bound in property["output"]:
            #file_object.write("Assert <= {} 1.0 x_{}_{}".format(network_length-1, bound["upper"],index))
            file_object.write("Assert <= {} 1.0 {}".format(bound["Lower"], network.layers[-1].nodes[index].name))
            file_object.write("\n")