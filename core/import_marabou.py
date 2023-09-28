import os
import sys

from core.configuration.consts import CODE_DIR

def dynamically_import_marabou(query_type="basic"):
    """
    dynamically import the relevant marabou version a.t. property type
    :param property_type: Str, type of the query
    """
    suffices_map = {
        "basic": "",
        "adversarial": "",
        "acas_xu_conjunction": "",
        # "basic": "_Reg",
        # "adversarial": "_Adv",
        # "acas_xu_conjunction": "_Adv"
    }
    MARABOU_DIR = os.path.join(CODE_DIR, f"Marabou{suffices_map[query_type]}")
    MARABOUPY_DIR = os.path.join(MARABOU_DIR, "maraboupy")
    sys.path.append(MARABOU_DIR)
    sys.path.append(MARABOUPY_DIR)

    # verity that the import works
    from maraboupy import MarabouCore
    from maraboupy import MarabouNetworkNNet as mnn
    print(f"sys.path={sys.path}")
    print("finish import marabou")

# if __name__ == "__main__":
#     dynamically_import_marabou()
