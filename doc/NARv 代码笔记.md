相比于 CEGAR_NN，abstraction 文件夹多了 after_abstraction、global、kmeans：

+ global：NARv 的最重要抽象方法。先预处理（do_process_before）为 inc / dec 节点抽象网络，然后计算节点 contribution，

pre_process 文件夹多了 calc_bounds、after_process、del_dec_nodes、generate_ori_map、pre_process_mine、read_input_bound、split_inc_dec_update：

+ pre_process_mine 作为根接口，包括 clac_bounds 和 del_dec_nodes 两种方法。

+ clac_bounds：ReluVal 计算节点的上下界。每个节点都有一个 name 字段，用 map 对应到具体的节点数据对象 name2node、序号 name2index 等。
+ del_dec_nodes：删除非激活节点。
+ split_inc_dec_update：update 预处理后的节点出边。

utils 文件夹多了 cal_contribution、cal_layer_average_in_weight、cluster、combine_influ_of2nodes、comjoin、dict_merge、find_relation、hierachy、network2rlv、propagation 文件。



# Narv.py

如果 mechanism 是 marabou_with_ar （158 行），那么需要选择 abstraction 方式 global，kmeans，complete，或 heuristic_alg2：

+ global: NARv 的最重要抽象方式，根据节点的 contribution 抽象。其中 propagation 用于处理被删除的节点。
+ kmeans: NARv 的抽象方式，根据 k_means 聚类节点抽象。
+ complete: CEGAR_NN 论文中描述的逐层抽象方式。
+ alg2: CEGAR_NN 论文中根据 clustering 抽象的算法 2。

当 sat == True 时，表示发现反例，但 sat == False 时表示还没有发现反例，不是结果是 UNSAT，结果是 query_result。（205 行）

218 行调用 Marabou 给出 query_result 的结果。

如果抽象后网络 query_result == SAT （234 行），则用 speedy_evaluate 在反例上赋值。如果在原网络输出小于零，则证明是假反例。否则再检查（is_satisfying_assignment）是否是满足性赋值，如果是，则确定是一个反例。否则它还是 spurious 的（269 行）。

既然找到的反例是 spurious 的，就需要 refine 了（284 行），refine 的方式有以下几种：

+ cegar：根据假反例 refine，返回 refine 后的 net。
+ global：NARv 的 refine 方式，返回一步 global refine 后的 net。

然后再用 is_satisfying_assignment 检查 spurious 反例，如果还是 False，则回到 218 行 Marabou 检查。

# Network.py

在 CEGAR_NN 中包含如下方法：

+ generate_weights、generate_bias、get_general_net_data、generate_name2node。
+ evaluate 和 speedy_evaluate：
+ remove_node：例如 merge 后需要将原有的被 merge 节点删去。
+ get_part2loss_map：



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