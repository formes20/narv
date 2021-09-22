import os
import time
from core.configuration.consts import (
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES, VERBOSE
)
from core.pre_process.pre_process import preprocess, preprocess_split_pos_neg
from core.pre_process.pre_process_mine import do_process_after, do_process_before
from core.nnet.read_nnet import network_from_nnet_file
from core.data_structures.Network import Network
from core.data_structures.Layer import Layer
from core.data_structures.ARNode import ARNode
from core.data_structures.Edge import Edge
from core.utils.verification_properties_utils import TEST_PROPERTY_ACAS
from core.abstraction.global_abstraction import global_abstraction_based_on_contribution
from core.abstraction.alg2 import heuristic_abstract_alg2
from core.utils.marabou_query_utils import reduce_property_to_basic_form
from one_experiment_global import one_experiment,generate_results_filename
import copy
import eventlet
def test_process(abstraction_type,refinement_type,property_id):
    # nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    #filename = "ACASXU_run2a_3_3_batch_2000.nnet"
    #nnet_filename = "ACASXU_run2a_1_1_batch_2000.nnet"
    # nnet_filename = os.path.join(nnet_dir, filename)
    # net1 = network_from_nnet_file(nnet_filename)
    # net2 = copy.deepcopy(net1)
    #print(net)
    #t1 = time.time()
    # do_process_before(net,"adversarial_0")
    # #print(net)
    # #print(net)
    # t2 = time.time()
    # preprocess(net)
    # t3 = time.time()
    # do_process_after(net)
    # t4 = time.time()
    
    # print("区间计算时间"+str(t2-t1))
    # print("分裂节点时间"+str(t3-t2))
    # print("删除节点时间"+str(t4-t3))
     
    eventlet.monkey_patch()
    for i in range(2,10):
        for j in range(2,10):
            with eventlet.Timeout(72000,False):
                filename = "ACASXU_run2a_"+str(i)+"_"+str(j)+"_batch_2000.nnet"
                one_experiment(filename,refinement_type,abstraction_type,"marabou_with_ar",1,100,"./result",property_id)
    #         nnet_filename = os.path.join(nnet_dir, filename)
    #         net1 = network_from_nnet_file(nnet_filename)
    #         #print(net1)
    #         net2 = copy.deepcopy(net1)
    #         test_property = TEST_PROPERTY_ACAS["basic_2"]
    # #net1,random_imputs = global_abstraction_based_on_contribution(net1,test_property)
    # #print(net1)
    # #property_dict = TEST_PROPERTY_ACAS["adversarial_0"]
    #         test_property2 = copy.deepcopy(test_property)
    #         net2, test_property2 = reduce_property_to_basic_form(network=net2, test_property=test_property2)
    #         net2 = heuristic_abstract_alg2(net2,test_property2)
    #print(net2)
    #print(net)
if __name__ == '__main__':
    test_process("complete","global","basic_2")