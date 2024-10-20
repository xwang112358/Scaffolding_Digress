from typing import Union

def generate_scaffolds_dict(smile_list):
    scaffolds = {}
    data_len = len(smile_list)
    for i in range(data_len):
        scaffold = _generate_scaffold(smile_list[i])
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [i]
            else:
                scaffolds[scaffold].append(i)

    # sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}

    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(),
            key=lambda x: (len(x[1]), x[1][0]),
            reverse=True
        )
    ]

    all_scaffold_sets_smiles = [
        (scaffold, scaffold_set) for (scaffold, scaffold_set) in sorted(
            scaffolds.items(),
            key=lambda x: (len(x[1]), x[1][0]),
            reverse=True
        )
    ]

    return all_scaffold_sets, all_scaffold_sets_smiles

def _generate_scaffold(smiles: str, include_chirality: bool = False) -> Union[str, None]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    if scaffold == '':
        return smiles

    return scaffold
