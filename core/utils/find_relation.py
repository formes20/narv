from core.data_structures.Abstract_action import Abstract_action
from core.utils.comjoin import is_subseq 

def find_relation(action_1, actions):
    if action_1.types == "delete":
        if len(action_1.name_1.split("+"))>1:
            for action_index in range(len(actions)-1,-1,-1):
                action = actions[action_index]
                if action.types == "combine" and action.layer == action_1.layer:
                    if is_subseq(action_1.name_1.split("+"),action.name_1.split("+")) and is_subseq(action_1.name_1.split("+"),action.name_2.split("+")):
                        for relyed in action.relyed:
                            relyed.rely.remove(action)
                        actions.remove(action)
    else:
        for action in actions:
            if action.types == "combine" and action.layer == action_1.layer:
                if is_subseq(action_1.name_1.split("+"),action.name_1.split("+")) and is_subseq(action_1.name_1.split("+"),action.name_2.split("+")):
                    action_1.rely.append(action)
                    action.relyed.append(action_1)
            if action.layer == action_1.layer - 1:
                action.relyed.append(action_1)
                action_1.rely.append(action)
        actions.append(action_1)
        print("num of actions")
        print(len(actions))
    return actions