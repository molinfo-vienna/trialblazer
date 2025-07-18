from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors

"""
Script to calculate physicochemical properties of molecules:
adopted from the script using for ring systems project from Ya Chen's work: https://github.com/anya-chen/RingSystems
"""


def get_physicochemical_properties(molDF, smiles_column) -> None:
    """Applies all property calculations to the ring systems of the dataframe and stores each property in a new column
    :param molDF: dataframe with ring systems as SMILES in the column 'ringSmiles'
    :return: a dataframe with ring system molecules and their properties.
    """
    PandasTools.AddMoleculeColumnToFrame(molDF, smiles_column, "Molecule")
    molDF["N"] = molDF["Molecule"].apply(get_molecule_composition, args=(7,))
    molDF["O"] = molDF["Molecule"].apply(get_molecule_composition, args=(8,))
    molDF["chiral"] = molDF["Molecule"].apply(get_nof_chiral_centers)
    molDF["MW"] = molDF["Molecule"].apply(get_MW)
    molDF["heavy_atoms"] = molDF["Molecule"].apply(num_heavy_atoms)
    molDF["h_acc"] = molDF["Molecule"].apply(
        num_of_h_acceptors_and_donors,
        args=(True,),
    )
    molDF["h_don"] = molDF["Molecule"].apply(
        num_of_h_acceptors_and_donors,
        args=(False,),
    )
    molDF["logP"] = molDF["Molecule"].apply(get_logp)
    molDF["TPSA"] = molDF["Molecule"].apply(get_TPSA)
    molDF["numAro"] = molDF["Molecule"].apply(num_aromatic_atoms)
    molDF["formalCharge"] = molDF["Molecule"].apply(sum_formal_charge)
    molDF["bridgeheadAtoms"] = molDF["Molecule"].apply(num_bridgehead_atoms)
    molDF["frac_csp3"] = molDF["Molecule"].apply(fraction_csp3)


def get_further_physicochemical_properties(molDF) -> None:
    """:param molDF: dataframe with ring systems as SMILES in the column 'ringSmiles'
    :return: a dataframe with ring system molecules and their properties
    """
    del molDF["bridgeheadAtoms"]
    molDF["S"] = molDF["Molecule"].apply(get_molecule_composition, args=(16,))
    molDF["nHalogens"] = molDF["Molecule"].apply(num_halogens)
    molDF["MR"] = molDF["Molecule"].apply(get_mr)


def get_molecule_composition(mol, requestedAtomicNum):
    """Counts the number of atoms of a given element in the ring system
    :param mol: the ring system molecule
    :param requestedAtomicNum: atomic number of the element for which the occurrence should be counted
    :return: the number of atoms of an element.
    """
    counter = 0
    for atom in mol.GetAtoms():
        atomicNum = atom.GetAtomicNum()
        if atomicNum == requestedAtomicNum:
            counter += 1
    return counter


def get_nof_chiral_centers(mol):
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))


def get_MW(mol):
    return round(Descriptors.MolWt(mol), 3)


def num_heavy_atoms(mol):
    return Lipinski.HeavyAtomCount(mol)


def num_of_h_acceptors_and_donors(mol, acc=True):
    if acc:
        return Lipinski.NumHAcceptors(mol)
    return Lipinski.NumHDonors(mol)


def get_logp(mol):
    return round(Crippen.MolLogP(mol), 3)


def get_TPSA(mol):
    return round(Descriptors.TPSA(mol), 3)


def num_aromatic_atoms(mol):
    numAromaticAtoms = 0
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            numAromaticAtoms += 1
    return numAromaticAtoms


def sum_formal_charge(mol):
    formalCharge = 0
    for atom in mol.GetAtoms():
        formalCharge += atom.GetFormalCharge()
    return formalCharge


def num_bridgehead_atoms(mol):
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol)


def fraction_csp3(mol):
    return round(Descriptors.FractionCSP3(mol), 3)


def num_halogens(mol):
    return Chem.Fragments.fr_halogen(mol)


def get_mr(mol):
    """Wildman-Crippen MR value
    Uses an atom-based scheme based on the values in the paper:
    Wildman and G. M. Crippen JCICS 39 868-873 (1999).
    """
    return round(Crippen.MolMR(mol), 3)
