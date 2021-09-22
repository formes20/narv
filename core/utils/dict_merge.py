from typing import TypeVar, Dict, Callable, List, AnyStr
AnyType = TypeVar("T")



def fun_dict_merge(dict1:Dict, dict2:Dict) -> Dict:
    if dict2:
        for key in dict2.keys():
            if key in dict1.keys():
                dict1[key] = dict2[key] + dict1[key]
            else:
                dict1[key] = dict2[key]
    return dict1
