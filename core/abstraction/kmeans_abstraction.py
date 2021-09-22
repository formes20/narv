from scipy.cluster import hierarchy
from core.abstraction.step import union_couple_of_nodes
from core.configuration.consts import VERBOSE, FIRST_ABSTRACT_LAYER
from core.data_structures.Network import Network
from core.pre_process.pre_process import preprocess,preprocess_updated
from core.pre_process.after_preprocess import after_preprocess
from core.utils.debug_utils import debug_print
from core.utils.abstraction_utils import finish_abstraction
from core.visualization.visualize_network import visualize_network
from core.pre_process.pre_process_mine import do_process_before,do_process_after
#from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.abstraction.step import union_couple_of_nodes
from core.utils.abstraction_utils import finish_abstraction
from core.configuration.consts import (
    VERBOSE, FIRST_ABSTRACT_LAYER, INT_MIN, INT_MAX
)
from core.utils.verification_properties_utils import TEST_PROPERTY_ACAS
from core.utils.alg2_utils import has_violation, get_limited_random_inputs
from core.utils.cal_contribution import calculate_contribution
from core.utils.cal_layer_average_in_weight import cal_average_in_weight
from core.utils.combine_influ_of2nodes import combin_influnce
from core.data_structures.Abstract_action import Abstract_action 
from core.utils.find_relation import find_relation
from core.abstraction.after_abstraction import after_abstraction
from core.utils.propagation import propagation_net
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.utils.ar_utils import calculate_weight_of_edge_between_two_part_groups
from core.utils.cluster import kMeans
from core.utils.hierarchy import hierarchy_cluster
import numpy as np
import copy
import time
import _pickle as cPickle
def kmeans_abstraction_based_on_contribution(network: Network,test_property:dict,do_preprocess:bool=True,sequence_length:int=50):
    network.generate_in_edge_weight()
    #print(network)6
    # layer_influence = []
    # for i in range(1,len(network.layers)-1):
    #     layer_influence.append(cal_average_in_weight(network, i))
    # print(layer_influence)
    actions = []
    # last abstract layer for adversarial properties
    last_free_layer = len(network.layers)-3 
    last_last_free_layer = last_free_layer
    sat = False
    t1 = time.time()
    do_process_before(network,test_property)
    t2 = time.time()
    print("bound analysis time {}".format(t2-t1))
    ###############only for adversarial properties#######################
    random_inputs = []
    print("output upper bound")
    print(network.layers[-1].nodes[0].upper_bound)
    if network.layers[-1].nodes[0].upper_bound < 0.0001:
        print(network.layers[-1].nodes[0].upper_bound)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXUNSAT,FOUND IN INTERVAL CALCULATIONXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        return network,random_inputs,network,sat,actions
    ###################   may introduce more interval computing methods later   ############
    layer_max_upperbound = []
    for i in range(FIRST_ABSTRACT_LAYER,len(network.layers)-1):
        max_upper = INT_MIN
        for node in network.layers[i].nodes:
            if node.upper_bound > max_upper:
                max_upper = node.upper_bound
        layer_max_upperbound.append(max_upper)
    #print(layer_max_upperbound)
    nodename2contribution_map = {}
    #influnce_arg = 1
    # t1 = time.time()
    do_process_after(network)
    preprocess_updated(network)
    # print(network)
    #print(network)
    #t2 = time.time()
    # print(t2-t1)
    #after_preprocess(network,False)
    #print(network)
    ############################################################################################################加代码#######################################################
    #do_process_after(network)
    #t3 = time.time()
    #print("del time")
    #print(t3-t2)
    features_inc, features_inc_name, features_dec, features_dec_name = generate_feature(network)
    # print("features_inc")
    # print(features_inc)
    cluster_each_layer(features_inc, features_inc_name,True)
    return
    preprocessed_network = cPickle.loads(cPickle.dumps(network,-1))
    preprocessed_network.generate_name2node_map()

    input_size = len(network.layers[0].nodes)
    # generate random inputs in the bound of the test property
    #property_dict = TEST_PROPERTY_ACAS[test_property]
    random_inputs = get_limited_random_inputs(
        input_size=input_size,
        test_property=test_property
    )
    #print(random_inputs)
    count = 0
    #print(random_inputs[0])
    for j in range(FIRST_ABSTRACT_LAYER,len(network.layers)-1):
    #    influnce_arg = influnce_arg * layer_influence[i]
        for node in network.layers[j].nodes:
            if layer_max_upperbound[j-FIRST_ABSTRACT_LAYER] == 0:
                return network,random_inputs,preprocessed_network,sat,actions
            node_contribution = node.cal_contri() / layer_max_upperbound[j-FIRST_ABSTRACT_LAYER]
            nodename2contribution_map[node.name] = node_contribution



    alert = 0
    while not has_violation(network, test_property, random_inputs):
    ####### loop #########
        #t0 = time.time()
        network.generate_in_edge_weight()
        network.generate_name2node_map()
        #nodes2edge_between_map = copy.deepcopy(network.get_nodes2edge_between_map())
        nodes2edge_between_map = network.get_nodes2edge_between_map()
        #print(nodes2edge_between_map)
        #print(nodename2contribution_map)
        if not nodename2contribution_map:
            break
        candicate_node = network.name2node_map[sorted(nodename2contribution_map.items(),key=lambda item:item[1])[0][0]]
        layer_index = int(candicate_node.name.split("_")[1])
        if layer_index == len(network.layers) - 2:
            alert +=1
            if alert == len(network.layers[layer_index].nodes) - 1:
                break
        #print(sorted(nodename2contribution_map.items(),key=lambda item:item[1]))
        #neg_inc = candicate_node.ar_type
        #print(nodename2contribution_map[candicate_node.name])
        combine_influ_list = []
        index = int(candicate_node.name.split("_")[1])
        for inode in network.layers[index].nodes:
            if inode != candicate_node:
                if  inode.name.split("_")[3] == candicate_node.name.split("_")[3].split('+')[0] and not inode.deleted:
                    combine_influ_list.append((inode.name,combin_influnce(inode, candicate_node, network, index, nodes2edge_between_map)))

        #print(sorted(combine_influ_list,key = lambda x:x[1])):
        if count == 0:
            network_before_propagation = preprocessed_network
            actions_last_time = cPickle.loads(cPickle.dumps(actions,-1))
        elif count % 30 == 0:
            network_before_propagation = cPickle.loads(cPickle.dumps(network,-1))
            
            #network_last_time = copy.deepcopy(network)

            #actions_last_time = copy.deepcopy(actions)
            actions_last_time = cPickle.loads(cPickle.dumps(actions,-1))
            last_last_free_layer = last_free_layer
        delete_inf = abs((candicate_node.upper_bound-candicate_node.lower_bound)/2)
        may_combine_item = None
        if combine_influ_list: 
            may_combine_item = sorted(combine_influ_list,key = lambda x:x[1])[0]
            # delete_inf = INT_MAX
            # if len(candicate_node.name.split("+")) == 1:
        if not may_combine_item or delete_inf <= may_combine_item[1]:
            #############delete##########
            print("delete a node")
            nodes_name = candicate_node.name.split("+")
            pre_layer_nodes = network.layers[layer_index-1].nodes
            next_layer_nodes = network.layers[layer_index+1].nodes
            for name in nodes_name:
                candi_node = preprocessed_network.name2node_map[name]
                if candicate_node.ar_type == "inc":
                    # print("inc")
                    bias = candi_node.upper_bound
                    # print("bias")
                    # print(bias)
                else:
                    # print("dec")
                    bias = candi_node.lower_bound
                    # print("bias")
                    # print(bias)
                part_node = ARNode(name=name,
                    ar_type=candicate_node.ar_type,
                    activation_func=candicate_node.activation_func,
                    in_edges=[],
                    out_edges=[],
                    bias=bias
                    )
                part_node.deleted = True

                for next_layer_node in next_layer_nodes:
                    group_a = name.split("+")
                    group_b = next_layer_node.name.split("+")
                    out_edge_weight = calculate_weight_of_edge_between_two_part_groups(
                        network=network, group_a=group_a, group_b=group_b)
                    out_edge = Edge(name,
                            next_layer_node.name,
                            out_edge_weight)
                    #next_layer_node = network.name2node_map[edge.dest]
                    part_node.out_edges.append(out_edge)
                    next_layer_node.in_edges.append(out_edge)
                for pre_layer_node in pre_layer_nodes:
                    in_edge = Edge(pre_layer_node.name,name,0)
                    #pre_layer_node = network.name2node_map[edge.src]
                    part_node.in_edges.append(in_edge)
                    pre_layer_node.out_edges.append(in_edge)
                network.layers[layer_index].nodes.append(part_node)   
                network.deleted_name2node[name] = cPickle.loads(cPickle.dumps(part_node,-1))
                
            action = Abstract_action("delete",candicate_node.name)
            actions = find_relation(action,actions)

            #network.deleted_name2node[candicate_node.name] = cPickle.loads(cPickle.dumps(candicate_node,-1))
            network.remove_node(candicate_node, index)
            network.biases = network.generate_biases()
            network.weights = network.generate_weights()
            print(candicate_node)
            print(nodename2contribution_map[candicate_node.name])
            print("delete operation")

            if last_free_layer > layer_index:
                last_free_layer = layer_index
            

            #network.deleted_nodename.append(candicate_node.name) #temp
            del nodename2contribution_map[candicate_node.name]

            count += 1
        else:
            union_name = "+".join([candicate_node.name, may_combine_item[0]])
            print(union_name)
            print("combine operation")
            action = Abstract_action("combine", candicate_node.name, may_combine_item[0])

            if last_free_layer > layer_index:
                last_free_layer = layer_index
            actions = find_relation(action,actions)

            union_couple_of_nodes(network, candicate_node, network.name2node_map[may_combine_item[0]])
            nl_p2u = {candicate_node.name: union_name, may_combine_item[0]: union_name}
            finish_abstraction(network=network,
                                next_layer_part2union=nl_p2u,
                                verbose=VERBOSE)
            nodename2contribution_map[union_name] = nodename2contribution_map[candicate_node.name] + nodename2contribution_map[may_combine_item[0]]
            del nodename2contribution_map[candicate_node.name]
            del nodename2contribution_map[may_combine_item[0]]
            count += 1
        
    if count == 0:
        sat = True
        print("times of operation:"+str(count))
        return network,random_inputs,preprocessed_network,sat,actions
    else:
        after_abstraction(network_before_propagation,last_last_free_layer,False)
        #print(network_last_time)
        print("times of operation:"+str(count))
        return network_before_propagation,random_inputs,preprocessed_network,sat,actions_last_time
        


def generate_feature(network: Network):
    features_inc = []
    features_dec = []
    features_inc_name = []
    features_dec_name = []
    for j in range(FIRST_ABSTRACT_LAYER,len(network.layers)-1):
        weights_inc = []
        weights_dec = []
        name_inc = []
        name_dec = []
        for node in network.layers[j].nodes:
            # map from name of node in previous layer to its index (in the previous layer)
            post_layer_name2index = {node.name: index for (index, node) in enumerate(network.layers[j+1].nodes)}
            layer_weights = []
            node_weights = [0.0] * len(network.layers[j+1].nodes)
            for out_edge in node.out_edges:
                dest_index = post_layer_name2index[out_edge.dest]
                node_weights[dest_index] = out_edge.weight
            if node.ar_type == "inc":
                weights_inc.append(node_weights)
                name_inc.append(node.name)
            else:
                weights_dec.append(node_weights)
                name_dec.append(node.name)
        features_inc.append(weights_inc)
        features_dec.append(weights_dec)
        features_inc_name.append(name_inc)
        features_dec_name.append(name_dec)
    return features_inc, features_inc_name, features_dec, features_dec_name

def cluster_each_layer(features, features_name, inc:bool):
    cluster2names_map = {}
    name2cluster_map = {}
    for i in range(len(features_name)):
        layer_index = features_name[0][0].split('_')[1]
        if inc:
            j = "inc"
        else:
            j = "dec"
        k = int(len(features[i])/2)   ## k 
        dataset = np.mat(features[i])
        print(dataset.shape)
        # centers, assignments = kMeans(dataset,k)
        num_clusters, indices = hierarchy_cluster(dataset, method='average', threshold=8.0)
        print ("%d clusters" % num_clusters)
        for k, ind in enumerate(indices):
            print ("cluster", k + 1, "is", ind)

        # print(assignments)
        # for index in range(len(assignments)):
        #     cluster_name = str(layer_index)+j+assignments[index][0]





