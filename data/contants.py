from rdkit import Chem
from rdkit.Chem import rdAbbreviations

# 官能团开始标志
RGroupBegin = ''

RGroupEnd = ''

BondTypeToIndexMap = {
    Chem.BondType.SINGLE: 1, 
    Chem.BondType.DOUBLE: 2, 
    Chem.BondType.TRIPLE: 3, 
    Chem.BondType.AROMATIC: 4, 
    Chem.BondType.IONIC: 5, 
    Chem.BondType.DATIVE: 6, 
}

RGroupSymbols = {
    'R': 8,
    'R<sub>1</sub>': 6,
    'R<sup>1</sup>': 2,
    'R<sub>2</sub>': 4,
    'R<sub>3</sub>': 4,
    'R<sub>4</sub>': 2,
    'R<sub>5</sub>': 2,
    'R<sub>6</sub>': 2,
    'R<sub>7</sub>': 1,
    'R<sub>8</sub>': 1,
    'R<sub>9</sub>': 1,
    'R<sub>10</sub>': 1,
    'R<sub>11</sub>': 1,
    'R<sub>12</sub>': 1,
    'R\'': 4,
    'R\'\'': 2,
    'R\'\'\'': 2,
    'R<sub>a</sub>': 4,
    'R<sub>b</sub>': 4,
    'R<sub>c</sub>': 4,
    'R<sub>d</sub>': 4,
    'A': 2,
    'Ar': 2,
    'X': 2,
    'Y': 2,
    'Z': 2
}
# 概率归一化
total = sum([v for v in RGroupSymbols.values()])
for k, v in RGroupSymbols.items():
    RGroupSymbols[k] = v / total

# https://github.com/syntelly/img2smiles_generator/blob/master/fgsmiles.py
outsides = [
    ['[CH3]', {'Me' : 0.9, 'CH3' : 0.1}, 0.811, 'C'],
    ['[CH2][CH3]', {'Et': 0.9, 'C2H5': 0.1}, 0.268, 'CC'],
    ['[CH2][CH2][CH3]', {'Pr': 0.9, 'C3H7': 0.1}, 0.083, 'CCC'],
    ['[CH1]([CH3])[CH3]', {'i-Pr': 0.9, 'iPr': 0.1}, 0.083, 'C(C)(C)'],
    ['[CH2][CH2][CH2][CH3]', {'Bu': 1}, 0.037, 'CCCC'],
    ['[CH2][CH1]([CH3])[CH3]', {'i-Bu': 0.9, 'iBu': 0.1}, 0.025, 'C(C(C)C)'],
    ['[CH1]([CH3])[CH2][CH3]', {'s-Bu': 0.9, 'sBu': 0.1}, 0.011, 'C(C)(CC)'],
    ['[CH0]([CH3])([CH3])[CH3]', {'t-Bu': 0.9, 'tBu': 0.1}, 0.043, 'C(C)(C)(C)'],
    ['O[CH3]', {'OMe': 0.9, 'OCH3': 0.1}, 0.182, 'O(C)'],
    ['O[CH2][CH3]', {'OEt': 0.9, 'OC2H5': 0.1}, 0.055, 'O(CC)'],
    ['O[CH2][CH2][CH3]', {'OPr': 0.9, 'OC3H7': 0.1}, 0.006, 'O(CCC)'],
    ['O[CH2][CH2][CH2][CH3]', {'OBu': 1}, 0.004, 'O(CCCC)'],
    ['[CH1](=O)', {'CHO': 1}, 0.011, 'C(=O)'],
    ['[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Ph': 1.0}, 0.127, 'c9ccccc9'],
    ['O[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OPh': 1.0}, 0.0064, 'O(c9ccccc9)'],
    ['[NH1][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHPh': 1.0}, 0.0045, 'N(c9ccccc9)'],
    ['[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'Tol': 0.8, 'p-Tol': 0.2}, 0.023, 'c9(ccc(C)cc9)'],
    ['S(=O)(=O)[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'Ts': 1}, 0.003, 'S(=O)(=O)(c9ccc(C)cc9)'],
    ['OS(=O)(=O)[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'OTs': 1}, 0.0001, 'O(S(=O)(=O)c9ccc(C)cc9)'],
    ['[NH1]S(=O)(=O)[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'NHTs': 1}, 0.001, 'N(S(=O)(=O)c9ccc(C)cc9)'],
    ['[CH0](=O)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Bz': 1}, 0.008, 'C(=O)(c9ccccc9)'],
    ['[NH1][CH0](=O)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHBz': 1}, 0.002, 'N(C(=O)c9ccccc9)'],
    ['[CH0](=[O])[CH3]', {'Ac': 1}, 0.029, 'C(=O)(C)'],
    ['O[CH0](=[O])[CH3]', {'OAc': 1}, 0.005, 'O(C(=O)C)'],
    ['[NH1][CH0](=[O])[CH3]', {'NHAc': 1}, 0.009, 'N(C(=O)C)'],
    ['[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Bn': 1}, 0.041, 'C(c9ccccc9)'],
    ['O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OBn': 1}, 0.009, 'O(Cc9ccccc9)'],
    ['[NH1][CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHBn': 1}, 0.004, 'N(Cc9ccccc9)'],
    ['[NH1][CH3]', {'NHMe': 1}, 0.322, 'N(C)'],
    ['[NH1][CH2][CH3]', {'NHEt': 1}, 0.399, 'N(CC)'],
    ['[NH1][CH2][CH2][CH3]', {'NHPr': 1}, 0.123, 'N(CCC)'],
    ['[NH1][CH2][CH2][CH2][CH3]', {'NHBu': 1}, 0.022, 'N(CCCC)'],
    ['[CH0](=O)O[CH0]([CH3])([CH3])[CH3]', {'Boc': 1}, 0.011, 'C(=O)(OC(C)(C)C)'],
    ['O[CH0](=O)O[CH0]([CH3])([CH3])[CH3]', {'OBoc': 1}, 0.0001, 'O(C(=O)OC(C)(C)C)'],
    ['[NH1][CH0](=O)O[CH0]([CH3])([CH3])[CH3]', {'NHBoc': 1}, 0.005, 'N(C(=O)OC(C)(C)C)'],
    ['[CH0](=O)O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Cbz': 1}, 0.003, 'C(=O)(OCc9ccccc9)'],
    ['O[CH0](=O)O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OCbz': 1}, 0.00001, 'O(C(=O)OCc9ccccc9)'],
    ['[NH1][CH0](=O)O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHCbz': 1}, 0.001, 'N(C(=O)OCc9ccccc9)'],
    ['[CH0](F)(F)F', {'CF3': 1}, 0.044, 'C(F)(F)(F)'],
    ['S(=O)(=O)C(F)(F)F', {'Tf': 1}, 0.0006, 'S(=O)(=O)(C(F)(F)F)'],
    ['OS(=O)(=O)C(F)(F)F', {'OTf': 1}, 0.0002, 'O(S(=O)(=O)C(F)(F)F)'],
    ['[CH0](=O)[CH0]([CH3])([CH3])[CH3]', {'Piv': 1}, 0.002, 'C(=O)(C(C)(C)C)'],
    ['O[CH0](=O)[CH0]([CH3])([CH3])[CH3]', {'OPiv': 1}, 0.0003, 'O(C(=O)C(C)(C)C)'],
    ['[CH1]=[CH2]', {'Vin': 0.5, 'Vi': 0.5}, 0.017, 'C=C'],
    ['[CH2][CH1]=[CH2]', {'All': 1}, 0.013, 'C(C=C)'],
    ['[SiH0]([CH3])([CH3])[CH3]', {'TMS': 0.5, 'SiMe3': 0.5}, 0.001, '[Si](C)(C)(C)'],
    ['O[SiH0]([CH3])([CH3])[CH3]', {'OTMS': 0.5, 'OSiMe3': 0.5}, 0.006, 'O([Si](C)(C)C)'],
    ['[SiH0]([CH3])([CH3])[CH0]([CH3])([CH3])[CH3]', {'TBS': 0.5, 'TBDMS': 0.5}, 0.002, '[Si]((C)(C)C(C)(C)C)'],
    ['O[SiH0]([CH3])([CH3])[CH0]([CH3])([CH3])[CH3]', {'OTBS': 0.5, 'OTBDMS': 0.5}, 0.002, 'O([Si](C)(C)C(C)(C)C)'],
    ['[CH1]1O[CH2][CH2][CH2][CH2]1', {'THP': 1}, 0.002, 'C9(OCCCC9)'],
    ['O[CH1]1O[CH2][CH2][CH2][CH2]1', {'OTHP': 1}, 0.0005, 'O(C9OCCCC9)'],
    ['[SiH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[CH0]([CH3])([CH3])[CH3]', {'TBDPS': 1}, 0.0002, '[Si]((c9ccccc9)(c9ccccc9)C(C)(C)C)'],
    ['O[SiH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[CH0]([CH3])([CH3])[CH3]', {'OTBDPS': 1}, 0.0002, 'O([Si](c9ccccc9)(c9ccccc9)C(C)(C)C)'],
    ['O[CH2]O[CH3]', {'OMOM': 1}, 0.0003, 'O(COC)'],
    ['[SiH0]([CH2][CH3])([CH2][CH3])[CH2][CH3]', {'TES': 0.5, 'SiEt3':0.5}, 0.0002, '[Si](CC)(CC)(CC)'],
    ['O[SiH0]([CH2][CH3])([CH2][CH3])[CH2][CH3]', {'OTES': 0.5, 'OSiEt3':0.5}, 0.0001, 'O([Si](CC)(CC)CC)'],
    ['[SiH0]([CH3])([CH3])[CH1]([CH3])[CH3]', {'IPDMS': 1}, 0.000001, '[Si]((C)(C)C(C)C)'],
    ['O[SiH0]([CH3])([CH3])[CH1]([CH3])[CH3]', {'OIPDMS': 1}, 0.00001, 'O([Si](C)(C)C(C)C)'],
    ['[SiH0]([CH2][CH3])([CH2][CH3])[CH1]([CH3])[CH3]', {'DEIPS': 1}, 0.00001, '[Si]((CC)(CC)C(C)C)'],
    ['O[SiH0]([CH2][CH3])([CH2][CH3])[CH1]([CH3])[CH3]', {'ODEIPS': 1}, 0.00001, 'O([Si](CC)(CC)C(C)C)'],
    ['[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', {'TIPS': 1}, 0.0001, '[Si]((C(C)(C))(C(C)(C))C(C)(C))'],
    ['O[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', {'OTIPS': 1}, 0.0001, 'O([Si](C(C)(C))(C(C)(C))C(C)(C))'],
    ['[CH1]1O[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])O[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])O[CH1]1', {'TIPDS': 1}, 0.00001, '[Si]((C(C)C)(C(C)C)C(C)C)'],
    ['[CH0](=O)C(F)(F)F', {'TFA': 1}, 0.002, 'C(=O)(C(F)(F)F)'],
    ['O[CH0](=O)C(F)(F)F', {'OTFA': 1}, 0.0006, 'O(C(=O)C(F)(F)F)'],
    ['[CH0](=O)O[CH2][CH1]1[cH0]2[cH1][cH1][cH1][cH1][cH0]2-[cH0]2[cH1][cH1][cH1][cH1][cH0]21', {'Fmoc': 1}, 0.0003, 'C(=O)(OCC8c9ccccc9-c9ccccc98)'],
    ['O[CH0](=O)O[CH2][CH1]1[cH0]2[cH1][cH1][cH1][cH1][cH0]2-[cH0]2[cH1][cH1][cH1][cH1][cH0]21', {'OFmoc': 1}, 0.00001, 'O(C(=O)OCC8c9ccccc9-c9ccccc98)'],
    ['[NH1][CH0](=O)O[CH2][CH1]1[cH0]2[cH1][cH1][cH1][cH1][cH0]2-[cH0]2[cH1][cH1][cH1][cH1][cH0]21', {'NHFmoc': 1}, 0.0003, 'N(C(=O)OCC8c9ccccc9-c9ccccc98)'],
    ['[CH0](=O)O[CH2][CH1]=[CH2]', {'Alloc': 1}, 0.0006, 'C(=O)(OCC=C)'],
    ['O[CH0](=O)O[CH2][CH1]=[CH2]', {'OAlloc': 1}, 0.00001, 'O(C(=O)OCC=C)'],
    ['[NH1][CH0](=O)O[CH2][CH1]=[CH2]', {'NHAlloc': 1}, 0.0002, 'N(C(=O)OCC=C)'],
    ['[CH0](=O)O[CH2][CH0](Cl)(Cl)Cl', {'Troc': 1}, 0.0001, 'C(=O)(OCC(Cl)(Cl)Cl)'],
    ['O[CH0](=O)O[CH2][CH0](Cl)(Cl)Cl', {'OTroc': 1}, 0.00001, 'O(C(=O)OCC(Cl)(Cl)Cl)'],
    ['[NH1][CH0](=O)O[CH2][CH0](Cl)(Cl)Cl', {'NHTroc': 1}, 0.00001, 'N(C(=O)OCC(Cl)(Cl)Cl)'],
    ['[CH0](=O)O[CH2][CH2][SiH0]([CH3])([CH3])[CH3]', {'Teoc': 1}, 0.00001, 'C(=O)(OCC[Si](C)(C)C)'],
    ['O[CH0](=O)O[CH2][CH2][SiH0]([CH3])([CH3])[CH3]', {'OTeoc': 1}, 0.00001, 'O(C(=O)OCC[Si](C)(C)C)'],
    ['[NH1][CH0](=O)O[CH2][CH2][SiH0]([CH3])([CH3])[CH3]', {'NHTeoc': 1}, 0.00001, 'N(C(=O)OCC[Si](C)(C)C)'],
    ['[CH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Tr': 1}, 0.08, 'C(c9ccccc9)(c9ccccc9)(c9ccccc9)'],
    ['O[CH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OTr': 1}, 0.08, 'O(C(c9ccccc9)(c9ccccc9)c9ccccc9)'],
    ['[NH1][CH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHTr': 1}, 0.08, 'N(C(c9ccccc9)(c9ccccc9)c9ccccc9)'],
    ['[CH0](=S)[NH0]([CH3])([CH3])', {'DMTC': 1}, 0.0001, 'C(=S)(N(C)C)'],
    ['O[CH0](=S)[NH0]([CH3])([CH3])', {'ODMTC': 1}, 0.00001, 'O(C(=S)N(C)C)'],
    ['[BH0]1O[CH0]([CH3])([CH3])[CH0]([CH3])([CH3])O1', {'BPin': 1}, 0.0003, 'B9(OC(C)(C)C(C)(C)O9)'],
    ['[CH0](=O)[CH2][CH2][CH0](=O)[CH3]', {'Lev': 1}, 0.0001, 'C(=O)(CCC(=O)C)'],
    ['O[CH0](=O)[CH2][CH2][CH0](=O)[CH3]', {'OLev': 1}, 0.00001, 'O(C(=O)CCC(=O)C)'],
    ['[cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'PMP': 1}, 0.0213, 'c9(ccc(OC)cc9)'],
    ['O[cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'OPMP': 1}, 0.0014, 'O(c9ccc(OC)cc9)'],
    ['[CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'PMB': 1}, 0.0053, 'C(c9ccc(OC)cc9)'],
    ['O[CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'OPMB': 1}, 0.0003, 'O(Cc9ccc(OC)cc9)'],
    ['[Sv2][CH3]', {'SMe': 1}, 0.0251, 'S(C)'],
    ['[Sv2][CH2][CH3]', {'SEt': 1}, 0.0044, 'S(CC)'],
    ['[Sv2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'SPh': 1}, 0.0018, 'S(c9ccccc9)'],
    ['N=[N+]=[N-]', {'N3': 1}, 0.0012, 'N(=[N+]=[N-])'],
    ['[cH0]1n[cH0]2[cH1][cH1][cH1][cH1][cH0]2s1', {'Bt': 1}, 0.0036, 'c8(nc9ccccc9s8)'],
    ['[N+](=O)[O-]', {'NO2': 1}, 0.0432, '[N+](=O)([O-])'],
    ['O[CH0](=O)[CH1](O[CH3])[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OMPA': 1}, 0.00001, 'O(C(=O)C(OC)c9ccccc9)'],
    ['[C]#[NH0]', {'CN': 1}, 0.5491, 'C(#N)'],
    ['[CH0](=O)[OH1]', {'CO2H': 0.5, 'COOH': 0.5}, 0.08, 'C(=O)(O)'],
    ['[CH0](=O)Cl', {'COCl': 1}, 0.001, 'C(=O)(Cl)'],
    ['[Sv6](=O)(=O)[CH3]', {'SO2Me': 1}, 0.0112, 'S(=O)(=O)(C)'],
    ['[Sv6](=O)(=O)[CH2][CH3]', {'SO2Et': 1}, 0.002, 'S(=O)(=O)(CC)'],
    ['[Sv6](=O)(=O)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'SO2Ph': 1}, 0.0026, 'S(=O)(=O)(c9ccccc9)'],
    ['[Sv6](=O)(=O)[OH1]', {'SO3H': 1}, 0.0019, 'S(=O)(=O)(O)'],
    ['[cH0]1[cH0]([CH3])[cH1][cH0]([CH3])[cH1][cH0]1[CH3]', {'Mes': 0.8, 'Ms': 0.2}, 0.0022, 'c9(c(C)cc(C)cc9C)'],
    ['[BH0]([OH])[OH]', {'B(OH)2': 1}, 0.0005, 'B(O)(O)'],
    ['[CH0](=O)O[CH3]', {'CO2Me': 0.5, 'COOMe': 0.5}, 0.003, 'C(=O)(OC)'],
    ['[CH0](=O)O[CH2][CH3]', {'CO2Et': 0.5, 'COOEt': 0.5}, 0.0234, 'C(=O)(OCC)'],
    ['[NH1][OH1]', {'NHOH': 1}, 0.001, 'N(O)'],
    ['[NH1][NH2]', {'NHNH2': 1}, 0.0069, 'NN'],
    ['[N]([CH3])[CH3]', {'NMe2': 1}, 0.0302, 'N((C)C)'],
    ['[N]([CH2][CH3])[CH2][CH3]', {'NEt2': 1}, 0.0081, 'N((CC)CC)'],
    ['[CH1]1[CH2][CH2][CH2][CH2][CH2]1', {'Cy': 1}, 0.0123, 'C9CCCCC9'],
    ]

# https://gist.github.com/greglandrum/c2df5e0ad91f4eee050848554264eff7
outside_str = '''Me C
MeO OC
MeS SC
MeN NC
CF CF
CF<sub>3</sub> C(F)(F)F
CN C#N
F<sub>3</sub>CN NC(F)(F)F
Ph c1ccccc1
NO N=O
NO<sub>2</sub> N(=O)=O
N(OH)CH<sub>3</sub> N(O)C
SO<sub>3</sub>H S(=O)(=O)O
COOH C(=O)O
nBu CCCC
EtO OCC
OiBu OCC(C)C
iPr CCC
tBu C(C)(C)C
Ac C(=O)C
AcO OC(=O)C
NHAc NC(=O)C
OR O*
#BzO OCc1ccccc1
BzO OC(=O)C1=CC=CC=C1
THPO O[C@@H]1OCCCC1
CHO C=O
NOH NO 
CO<sub>2</sub>Et C(=O)OCC
CO<sub>2</sub>Me C(=O)OC
MeO<sub>2</sub>S S(=O)(=O)C
NMe<sub>2</sub> N(C)C
CO<sub>2</sub>R C(=O)O*
ZNH NC(=O)OCC1=CC=CC=C1
HOCH<sub>2</sub> CO
H<sub>2</sub>NCH<sub>2</sub> CN
Et CC
BnO OCC1=CC=CC=C1
AmNH NCCCCC
AmO OCCCCC
AmO<sub>2</sub>C C(=O)OCCCCC
AmS SCCCCC
BnNH NCC1=CC=CC=C1
BnO<sub>2</sub>C C(=O)OCC1=CC=CC=C1
Bu<sub>3</sub>Sn [Sn](CCCC)(CCCC)CCCC
BuNH NCCCC
BuO OCCCC
BuO<sub>2</sub>C C(=O)OCCCC
BuS SCCCC
CBr<sub>3</sub> C(Br)(Br)Br
CbzNH NC(=O)OCC1=CC=CC=C1
CCl<sub>3</sub> C(Cl)(Cl)Cl
ClSO<sub>2</sub> S(=O)(=O)Cl
COBr C(=O)Br
COBu C(=O)CCCC
COCF<sub>3</sub> C(=O)C(F)(F)F
COCl C(=O)Cl
COCO C(=O)C=O
COEt C(=O)CC
COF C(=O)F
COMe C(=O)C
OCOMe OC(=O)C
CONH<sub>2</sub> C(=O)N
CONHEt C(=O)NCC
CONHMe C(=O)NC
COSH C(=O)S
Et<sub>2</sub>N N(CC)CC
Et<sub>3</sub>N N(CC)(CC)CC
EtNH NCC
H<sub>2</sub>NSO<sub>2</sub> S(=O)(N)=O
HONH ON
Me<sub>2</sub>N N(C)C
NCO N=C=O
NCS N=C=S
NHAm NCCCCC
NHBn NCC1=CC=CC=C1
NHBu NCCCC
NHEt NCC
NHOH NO
NHPr NCCC
NO N=O
POEt<sub>2</sub> P(OCC)OCC
POEt<sub>3</sub> P(OCC)(OCC)OCC
POOEt<sub>2</sub> P(=O)(OCC)OCC
PrNH CCCN
SEt SCC
BOC C(=O)OC(C)(C)C
MsO OS(=O)(=O)C
OTos OS(=O)(=O)c1ccc(C)cc1
Tos S(=O)(=O)c1ccc(C)cc1
C<sub>8</sub>H CCCCCCCC
C<sub>6</sub>H CCCCCC
CH<sub>2</sub>CH<sub>3</sub> CC
N(CH<sub>2</sub>CH3)<sub>2</sub> N(CC)CC
N(CH<sub>2</sub>CH<sub>2</sub>CH<sub>3</sub>)<sub>2</sub> N(CCC)CCC
C(CH<sub>3</sub>)<sub>3</sub> C(C)(C)C
COCH<sub>3</sub> C(=O)C
CH(CH<sub>3</sub>)<sub>2</sub> C(C)C
OCF<sub>3</sub> OC(F)(F)F
OCCl<sub>3</sub> OC(Cl)(Cl)Cl
OCF<sub>2</sub>H OC(F)F
SO<sub>2</sub>Me S(=O)(=O)C
OCH<sub>2</sub>CO<sub>2</sub>H OCC(=O)O
OCH<sub>2</sub>CO<sub>2</sub>Et OCC(=O)OCC
BOC<sub>2</sub>N N(C(=O)OC(C)(C)C)C(=O)OC(C)(C)C
BOCHN NC(=O)OC(C)(C)C
NHCbz NC(=O)OCc1ccccc1
OCH<sub>2</sub>CF<sub>3</sub> OCC(F)(F)F
NHSO<sub>2</sub>BU NS(=O)(=O)CCCC
NHSO<sub>2</sub>Me NS(=O)(=O)C
MeO<sub>2</sub>SO OS(=O)(=O)C
NHCOOEt NC(=O)OCC
NHCH<sub>3</sub> NC
H<sub>4</sub>NOOC C(=O)ON
C<sub>3</sub>H<sub>7</sub> CCC
C<sub>2</sub>H<sub>5</sub> CC
NHNH<sub>2</sub> NN
OCH<sub>2</sub>CH<sub>2</sub>OH OCCO
OCH<sub>2</sub>CHOHCH<sub>2</sub>OH OCC(O)CO
OCH<sub>2</sub>CHOHCH<sub>2</sub>NH OCC(O)CN
NHNHCOCH<sub>3</sub> NNC(=O)C
NHNHCOCF<sub>3</sub> NNC(=O)C(F)(F)F
NHCOCF<sub>3</sub> NC(=O)C(F)(F)F
CO<sub>2</sub>CysPr C(=O)ON[C@H](CS)C(=O)CCC
HOH<sub>2</sub>C CO
H<sub>3</sub>CHN NC
H<sub>3</sub>CO<sub>2</sub>C C(=O)OC
CF<sub>3</sub>CH<sub>2</sub> CC(F)(F)F
OBOC OC(=O)OC(C)(C)C
Bn<sub>2</sub>N N(Cc1ccccc1)Cc1ccccc1
F5S S(F)(F)(F)(F)F
PPh<sub>2</sub> P(c1ccccc1)c1ccccc1
PPh<sub>3</sub> P(c1ccccc1)(c1ccccc1)c1ccccc1
OCH<sub>2</sub>Ph OCc1ccccc1
CH<sub>2</sub>OMe COC
PMBN NCc1ccc(OC)cc1
SO<sub>2</sub> S(=O)=O
NH<sub>3</sub>Cl NCl
CF<sub>2</sub>CF<sub>3</sub> C(F)(F)C(F)(F)F
CF<sub>2</sub>CF<sub>2</sub>H C(F)(F)C(F)(F)
Bn Cc1ccccc1
OCH<sub>2</sub>Ph OCc1ccccc1
COOCH<sub>2</sub>Ph C(=O)OCc1ccccc1
Ph<sub>3</sub>CO OC(c1ccccc1)(c1ccccc1)c1ccccc1
Ph<sub>3</sub>C C(c1ccccc1)(c1ccccc1)c1ccccc1
Me<sub>2</sub>NO2S S(C)(C)N(=O)=O
SO<sub>3</sub>Na S(=O)(=O)(=O)[Na]
OSO<sub>2</sub>Ph OS(=O)(=O)c1ccccc1
(CH<sub>2</sub>)<sub>5</sub>Br CCCCCBr
OPh Oc1ccccc1
SPh Sc1ccccc1
NHPh Nc1ccccc1
CONEt<sub>2</sub> C(=O)N(CC)CC
CONMe<sub>2</sub> C(=O)N(C)C
EtO<sub>2</sub>CHN NC(=O)OCC
H<sub>4</sub>NO<sub>3</sub>S S(=O)(=O)ON
TMS [Si](C)(C)(C)
COCOOCH<sub>2</sub>CH<sub>3</sub> C(=O)C(=O)OCC
OCH<sub>2</sub>CN OCC#N'''

Abbreviations = {}
defaults = rdAbbreviations.GetDefaultAbbreviations()
# 解析默认的缩写
for i in defaults:
    smarts = Chem.MolToSmarts(i.mol)
    label = i.displayLabel
    label_w = i.displayLabelW
    if smarts not in Abbreviations:
        Abbreviations[smarts] = {}
    if label not in Abbreviations[smarts]:
        Abbreviations[smarts][label] = 0.7
    if label_w != '' and label_w not in Abbreviations[smarts]:
        Abbreviations[smarts][label_w] = 0.3
# 归一化
for k in Abbreviations.keys():
    total = sum(Abbreviations[k].values())
    for kk in Abbreviations[k].keys():
        Abbreviations[k][kk] /= total
# 加入外部的缩写
for i in outsides:
    smarts = i[0]
    if smarts not in Abbreviations:
        Abbreviations[smarts] = {}
    for k, v in i[1].items():
        if k not in Abbreviations[smarts]:
            Abbreviations[smarts][k] = v
# 再次归一化
for k in Abbreviations.keys():
    total = sum(Abbreviations[k].values())
    for kk in Abbreviations[k].keys():
        Abbreviations[k][kk] /= total
# 加入外部的缩写
for i in outside_str.split('\n'):
    label, smarts = i.strip().split(' ')
    if smarts not in Abbreviations:
        Abbreviations[smarts] = {}
    Abbreviations[smarts][label] = 1
# 再次归一化
for k in Abbreviations.keys():
    total = sum(Abbreviations[k].values())
    for kk in Abbreviations[k].keys():
        Abbreviations[k][kk] /= total