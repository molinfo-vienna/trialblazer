import os
from pathlib import Path
from rdkit import Chem

"""
Script to split multi-component compounds into individual compounds. Use Chem.GetMolFrags(mol,asMols=True).
"""


def separate_multicomponents_test(inputDir) -> None:
    outdir = Path(inputDir).parent / "separate_multicom_GetMolFrags"
    if not outdir.exists():
        outdir.mkdir()
    count = 0
    counter = 0
    for filename in os.listdir(inputDir):
        outfilename = filename.split(".")[0] + "_noMulticom.csv"
        infile = inputDir / filename
        outfile = outdir / outfilename
        with open(infile) as inputSmilesCSV:
            with open(outfile, "w") as of:
                header = inputSmilesCSV.readline()
                of.write(header)
                for line in inputSmilesCSV:
                    entry = line.rstrip().split("\t")
                    if len(entry) == 4:
                        preprocessedSmile = entry[1]
                        name = entry[2]
                        tautomerizedSmiles = entry[3]

                        pre_mol = Chem.MolFromSmiles(preprocessedSmile)
                        tauto_mol = Chem.MolFromSmiles(tautomerizedSmiles)

                        molFrag_pre = Chem.GetMolFrags(pre_mol, asMols=True)
                        if tauto_mol is None:
                            pass
                        else:
                            molFrag_tauto = Chem.GetMolFrags(
                                tauto_mol,
                                asMols=True,
                            )

                        smilesList = [Chem.MolToSmiles(mol) for mol in molFrag_pre]
                        smilesList2 = [Chem.MolToSmiles(mols) for mols in molFrag_tauto]

                        if len(smilesList) == 1:
                            of.write(line)
                        else:
                            count += 1
                            for i in range(len(smilesList)):
                                new_name = str(name) + "x" + str(i)
                                of.write(
                                    entry[0]
                                    + "\t"
                                    + smilesList[i]
                                    + "\t"
                                    + new_name
                                    + "\t"
                                    + smilesList2[i]
                                    + "\n",
                                )
                    else:
                        counter += 1
