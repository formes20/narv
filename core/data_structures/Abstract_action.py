from typing import TypeVar, Dict, Callable, List, AnyStr
from core.utils.comjoin import conjunction, join_atoms, is_subseq
import copy


class Abstract_action:

    def __init__(self, types, name_1, name_2=None):
        self.types = types  # allow three types ["combine","delete","delete_one"]
        if types == "delete_one":
            assert not name_2
        self.name_1 = name_1
        self.name_2 = name_2
        self.rely = []
        self.relyed = []
        self.layer = int(name_1.split('_')[1])
        # self.pos = self.name_1.split('_')[3]=="pos"
        self.inc = self.name_1.split('_')[3] == "inc"

    # def find_relation(self, actions:List)-> List:
    #     if self.types == "delete_one":
    #         temp = copy.deepcopy(self)
    #         actions.append(temp)
    #
    #     elif self.types == "delete":
    #         for action in actions:
    #             if action.types == "combine" and action.layer == self.layer:
    #                 if is_subseq(self.name_1.split("+"),action.name_1.split("+")) and is_subseq(
    #                         self.name_1.split("+"),action.name_2.split("+")):
    #                     self.rely.append(action)
    #                     action.relyed.append(self)
    #         temp = copy.deepcopy(self)
    #         actions.append(temp)
    #
    #     else:
    #         for action in actions:
    #             if action.types != "delete_one":
    #                 if (action.layer == self.layer + 1) or (action.layer == self.layer - 1):
    #                     action.relyed.append(self)
    #                     self.rely.append(action)
    #         temp = copy.deepcopy(self)
    #         actions.append(temp)
    #     return actions

    def refineable(self) -> bool:
        if not self.relyed:
            return True
        else:
            return False
