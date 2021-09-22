def conjunction(list_1, list_2) -> bool:
    assert list_1
    assert list_2
    a = list_1
    b = set(list_2) 
    c =[i for i, item in enumerate(a) if item in b]
    return bool(c)

def is_subseq(list_1,list_2):
    flag = True
    for b in list_2:
        if b not in list_1:
            flag = False
    return flag

def join_atoms(atoms, node_names, operation_names):
    parts = []
    new = True
    for operation_name in operation_names:
        if operation_name in node_names:
            parts.append(operation_name)
    if not atoms:
        atoms.append(parts)
    else:
        for atom in atoms:     
            if is_subseq(atom, parts):
                atoms.remove(atom)
                atoms.append(parts)
                new = False
            elif conjunction(atom,parts):
                new = False
        if new:
            atoms.append(parts)
    return atoms    