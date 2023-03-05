import io
import re
import random
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdAbbreviations

import torch
from torchvision import transforms
from torch.nn import functional as F
from skimage.util import random_noise
from torch.utils.data import Dataset, DataLoader
from data.contants import BondTypeToIndexMap, RGroupSymbols, RGroupBegin, RGroupEnd, Abbreviations


def to_canonical(smiles):
    # 去除原子电性，编号等信息用于计算准确率
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return '* |$Failed$|'
    for atom in mol.GetAtoms():
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)
        atom.SetAtomMapNum(0)
        for name in atom.GetPropNames():
            atom.ClearProp(name)
    order = Chem.CanonicalRankAtoms(mol, includeChirality=True)
    mol_ordered = Chem.RenumberAtoms(mol, list(order))
    smiles = Chem.MolToSmiles(mol_ordered)
    return smiles

def smiles_to_graph(smiles, *args, **kwargs):
    # 根据MIT论文将SMILES转换成原子和键构成的图 Robust Molecular Image Recognition: A Graph Generation Approach
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    atoms_num = len(atoms)
    bonds_graph = np.zeros((atoms_num, atoms_num), dtype=np.int64)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin > end:
            begin, end = end, begin
        bonds_graph[begin, end] = BondTypeToIndexMap[bond.GetBondType()]
    # 只取元素，名字不重要
    # 没有电性，后续考虑通过atom.SetFormalCharge(1)加上
    atoms_str = [a.GetSymbol() for a in atoms]
    bonds_mask = np.ones_like(bonds_graph, dtype=np.float32)
    # a.GetSmarts()
    return atoms_str, bonds_graph, bonds_mask

def graph_to_smiles(atoms, bonds_graph, *args, **kwargs):
    # 根据模型预估的图重构分子
    try:
        atoms = atoms.split(' ')
        if len(atoms) == 0:
            return '* |$Failed$|'
        # 加入所有原子
        mol = Chem.RWMol()
        for atom_symbol in atoms:
            atom = Chem.Atom(atom_symbol)
            mol.AddAtom(atom)
        # 根据图加入原子之间的键
        index_to_bond_type_map = {v: k for k, v in BondTypeToIndexMap.items()}
        num = len(bonds_graph)
        for i in range(num):
            for j in range(i+1, num):
                if bonds_graph[i, j] == 0:
                    continue
                bond_type = index_to_bond_type_map[bonds_graph[i, j]]
                mol.AddBond(i, j, bond_type)
        mol.GetMol()
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print('Error: graph_to_smiles\n', e, '\n', atoms, '\n', bonds_graph)
        return '* |$Failed$|'

def cxsmiles_remove_useless_info(cxsmiles):
    # 去除生成的CXSmiles中的dummyLabel等无用信息，用于缩短text输入输出，保留R基团名字
    suffix_pattern = '\|.*?\|'
    suffix = re.search(suffix_pattern, cxsmiles)
    if suffix is None:
        return cxsmiles

    prefix = cxsmiles[: suffix.start()-1]
    suffix = cxsmiles[suffix.start():]

    r_group_pattern = '\$.*?\$'
    r_group_info = re.search(r_group_pattern, suffix)
    if r_group_info is None:
        return prefix

    r_group_info = suffix[r_group_info.start(): r_group_info.end()]
    return '{} |{}|'.format(prefix, r_group_info)

def cxsmiles_encode(cxsmiles, smiles_tokenizer):
    # 将复杂的CXSmiles转换成更简单的FGSmiles，和这篇类似Image2SMILES: Transformer-Based Molecular Optical Recognition Engine
    if '*' not in cxsmiles or ' |$' not in cxsmiles:
        return cxsmiles
    cxsmiles = cxsmiles_remove_useless_info(cxsmiles)
    smiles = cxsmiles.split(' |$')
    # 取出Smiles和对应的R基团名字
    prefix, suffix = smiles
    suffix = suffix.split('$')[0]
    suffix = suffix.split(';')
    smiles_tokens = smiles_tokenizer.tokenize(prefix)
    mol = Chem.MolFromSmiles(cxsmiles)
    all_atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    out = []
    r_group_num = 0
    index = 0
    # 将Smiles中的*替换为对应的名字
    while index < len(smiles_tokens):
        t = smiles_tokens[index]
        if t.upper() not in all_atom_symbols and t.lower() not in all_atom_symbols and t not in all_atom_symbols:
            out.append(t)
        else:
            if t == '*':
                if suffix[r_group_num] != '':
                    if len(out) > 0 and out[-1] == '[':
                        out.pop(-1)
                        out.append('{}{}{}'.format(RGroupBegin, suffix[r_group_num], RGroupEnd))
                        while smiles_tokens[index] != ']':
                            index += 1
                    else:
                        out.append('{}{}{}'.format(RGroupBegin, suffix[r_group_num], RGroupEnd))
                else:
                    out.append('{}{}'.format(RGroupBegin, RGroupEnd))
            else:
                out.append(t)
            r_group_num += 1
        index += 1
    return ''.join(out)

def cxsmiles_decode(string, smiles_tokenizer):
    # 根据FGSmiles解码成CXSmiles，用于可视化计算准确率等
    contain_r_group = False
    for k in RGroupSymbols:
        if k in string:
            contain_r_group = True
            break
    if not contain_r_group:
        return string
    smiles_tokens = smiles_tokenizer.tokenize(string)
    r_group_names = []
    replace_smiles_token = []
    for token in smiles_tokens:
        if token in RGroupSymbols:
            r_group_names.append(token)
            replace_smiles_token.append('*')
        else:
            replace_smiles_token.append(token)

    smiles = ''.join(replace_smiles_token)
    suffix = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Error: cxsmiles_decode\n', string, '\n', smiles)
        return '* |$Failed$|'
    num = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            suffix.append(r_group_names[num]+';')
            num += 1
        else:
            suffix.append(';')
    suffix = ''.join(suffix)
    if suffix[-1] == ';':
        suffix = suffix[:-1]
    cxsmiles = '{} |${}$|'.format(smiles, suffix)
    return cxsmiles

def cxsmiles_to_graph(cxsmiles, smiles_tokenizer):
    # CXSmiles转换成图，相比于Smiles主要是对R基团的名字需要更复杂的处理
    # 最初的设计时 C[PAD]R group name[CLS]，通过加入[PAD][CLS]表示名字的开始结束，可以接受任意长度的R group name
    # 这样设计后一个名字长度不再一定是1，因此在bond graph需要一些特殊处理，只用最后的[CLS]表示这个基团的键
    # 但是这样太复杂并且没有必要，因此后续改为通过tokenizer加入特殊字符保持基团名字长度依然为1
    mol = Chem.MolFromSmiles(cxsmiles)
    atoms = mol.GetAtoms()
    seq_len = 0
    atom_names = []
    atom_to_name_map = {}
    bonds_mask = []
    for i, atom in enumerate(atoms):
        symbol = atom.GetSymbol()
        if symbol == '*':
            if atom.HasProp('atomLabel'):
                name = atom.GetProp('atomLabel')
            else:
                name = ''
            atom_names.append(symbol)
        else:
            atom_names.append(symbol)
        atom_to_name_map[i] = seq_len
        seq_len += 1
        bonds_mask += [1]
    bonds_graph = np.zeros((seq_len, seq_len), dtype=np.int64)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin > end:
            begin, end = end, begin
        bonds_graph[atom_to_name_map[begin], atom_to_name_map[end]] = BondTypeToIndexMap[bond.GetBondType()]
    bonds_mask = np.asarray(bonds_mask)
    bonds_mask = np.expand_dims(bonds_mask, axis=-1) * np.expand_dims(bonds_mask, axis=0)
    return atom_names, bonds_graph, bonds_mask

def graph_to_cxsmiles(atoms, bonds_graph, smiles_tokenizer):
    # graph转换为CXSmiles
    try:
        tokens = smiles_tokenizer.tokenize(atoms)
        if len(tokens) == 0:
            return '* |$Failed$|'
        mol = Chem.RWMol()
        name_to_atom_map = {}
        num = 0
        for i, t in enumerate(tokens):
            if t in RGroupSymbols:
                print(t)
                atom = Chem.Atom('*')
                atom.SetProp('atomLabel', t)
                mol.AddAtom(atom)
                name_to_atom_map[i] = num
                num += 1
            else:
                atom = Chem.Atom(t)
                mol.AddAtom(atom)
                name_to_atom_map[i] = num
                num += 1
        index_to_bond_type_map = {v: k for k, v in BondTypeToIndexMap.items()}
        num = len(bonds_graph)
        for i in range(num):
            for j in range(num):
                if bonds_graph[i, j] == 0:
                    continue
                if i >= j:
                    continue
                bond_type = index_to_bond_type_map[bonds_graph[i, j]]
                mol.AddBond(name_to_atom_map[i], name_to_atom_map[j], bond_type)
        mol.GetMol()
        smiles = Chem.MolToCXSmiles(mol)
        return smiles
    except Exception as e:
        print('Error: graph_to_cxsmiles\n', e, '\n', atoms, '\n', bonds_graph)
        return '* |$Failed$|'

def cut_molecule_and_replace_to_r_group(raw_smiles, num_r_groups=0):
    # 将一个已有的分子选择可切断部分替换为R基团
    if num_r_groups == 0:
        return raw_smiles, []
    # 解析smiles
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        print('Error: cut_molecule_and_replace_to_r_group 1\n', raw_smiles)
        return raw_smiles, []
    atom_num = len(mol.GetAtoms())
    # 确定R名称
    r_symbol = random.choices(list(RGroupSymbols.keys()), weights=list(RGroupSymbols.values()), k=1)[0]
    smiles = raw_smiles.split(' |$')
    # 解析CXSmiles
    assert(len(smiles) <= 2)
    if len(smiles) == 1:
        smiles_prefix = smiles[0]
        smiles_suffix = '$|'
    else:
        smiles_prefix, smiles_suffix = smiles
    # 加入两个R基团
    smiles_with_r = '[*].[*].{} |${};{};{}'.format(smiles_prefix, r_symbol, r_symbol, smiles_suffix)
    mol_with_r = Chem.MolFromSmiles(smiles_with_r)
    if mol_with_r is None:
        smiles_with_r_remove_suffix = cxsmiles_remove_useless_info(smiles_with_r)
        mol_with_r = Chem.MolFromSmiles(smiles_with_r_remove_suffix)
    if mol_with_r is None:
        print('Error: cut_molecule_and_replace_to_r_group 2\n', raw_smiles, '\n', smiles_with_r)
        return raw_smiles, []
    # 可修改分子
    emol = Chem.RWMol(mol_with_r)
    all_bond_num = len(emol.GetBonds())
    # 找到不在环里的原子
    r = emol.GetRingInfo()
    bonds_in_ring = set()
    for r in r.BondRings():
        for i in r:
            bonds_in_ring.add(i)
    # 不在环里即可以一刀切开
    # bonds_not_in_ring = [i for i in range(all_bond_num) if i not in bonds_in_ring]
    # 不切开已经是R的连接
    bond_index_from_r = set()
    for atom in emol.GetAtoms():
        if '*' in atom.GetSymbol():
            for bond in atom.GetBonds():
                bond_index_from_r.add(bond.GetIdx())
    valid_bonds = [i for i in range(all_bond_num) if i not in bonds_in_ring and i not in bond_index_from_r]
    if len(valid_bonds) == 0:
        return raw_smiles, []
    # 随机选择切开的位置
    bond_index_to_cut = random.choice(valid_bonds)
    bond_to_cut = emol.GetBondWithIdx(bond_index_to_cut)
    bond_to_cut_type = bond_to_cut.GetBondType()
    begin, end = bond_to_cut.GetBeginAtomIdx(), bond_to_cut.GetEndAtomIdx()
    # 切开
    emol.RemoveBond(begin, end)
    # 连接两个R
    emol.AddBond(0, begin, bond_to_cut_type)
    emol.AddBond(1, end, bond_to_cut_type)
    # 切开分子
    fragments = Chem.GetMolFrags(emol, asMols=True)
    main, sub = fragments
    # 选择原子多的作为主体
    if len(sub.GetAtoms()) > len(main.GetAtoms()):
        main, sub = sub, main
    # 次要部分去掉R基团
    sub = Chem.RWMol(sub)
    # 不确定是否一定在0
    sub.RemoveAtom(0)
    # 导出为smiles
    main = Chem.MolToCXSmiles(main)
    sub = (r_symbol, Chem.MolToCXSmiles(sub))
    # 有多个R时进行递归
    res = cut_molecule_and_replace_to_r_group(main, num_r_groups-1)
    return res[0], [sub]+res[1]

def molecule_abbrevs(mol):
    # 分子进行缩写
    abbrevs_string = []
    for k, v in Abbreviations.items():
        labels = list(v.keys())
        p = list(v.values())
        label = random.choices(labels, p, k=1)[0]
        abbrevs_string.append('{}    {}'.format(label, k))
    abbrevs_string = '\n'.join(abbrevs_string)
    abbrevs = rdAbbreviations.ParseAbbreviations(abbrevs_string)
    # abbrevs = rdAbbreviations.GetDefaultAbbreviations()
    abbrevsed_mol = rdAbbreviations.CondenseMolAbbreviations(mol, abbrevs)
    return abbrevsed_mol

class Image2SmilesDataset(Dataset):
    def __init__(self, smiles, transform=None):
        super().__init__()
        self.smiles = smiles
        self.transform = transform
        
    def __len__(self):
        return len(self.smiles) * 10000
    
    def __getitem__(self, idx):
        raw_smiles = self.smiles[idx % len(self.smiles)]
        data = self.transform(raw_smiles)
        data['raw_smiles'] = raw_smiles
        return data

class ImageTransform:
    def __init__(self, transforms=None, processor=None):
        self.transforms = transforms
        self.processor = processor
        
    def __call__(self, x):
        if self.transforms:
            x = self.transforms(x)
        if self.processor:
            x = self.processor(x, return_tensors='pt')['pixel_values'][0]
        return x

class SmilesRemoveNum:
    def __init__(self, p=0):
        self.p = p

    def __call__(self, smiles):
        if 1 - self.p >= random.random():
            return smiles
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(mol)
        return smiles

class SmilesReplaceWithR:
    def __init__(self, p=0, max_r_groups=6):
        self.p = p
        self.max_r_groups = max_r_groups

    def __call__(self, smiles):
        if 1 - self.p >= random.random():
            return smiles
        main, _ = cut_molecule_and_replace_to_r_group(smiles, self.max_r_groups)
        return main
    
class RandomTransform:
    def __init__(self, transform, p):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if self.p > random.random():
            return self.transform(x)
        else:
            return x

class RandomNoise:
    # 随机加入椒盐噪声
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if self.p > random.random():
            x = np.asarray(x)
            noise = np.random.random(x.shape)
            mask = (noise > self.p).astype(x.dtype)
            x = x * mask + 255 * (1-mask)
            x = x.astype(np.uint8)
            x = Image.fromarray(x)
            return x
        else:
            return x

class TransformV1:
    def __init__(self, smiles_tokenizer, smiles_transform=None, mol_transform=None, 
    image_transform=None, mol_size=(224, 224), return_graph=False, num_r_groups_in_text=0):
        self.smiles_tokenizer = smiles_tokenizer
        # 包括数据预处理和数据增强
        self.image_transform = image_transform
        # 包括去掉编号，压缩，随机变成R
        self.smiles_transform = smiles_transform
        # 分子变换，比如压缩
        self.mol_transform = mol_transform
        self.mol_size = mol_size
        self.return_graph = return_graph
        self.num_r_groups_in_text = num_r_groups_in_text

    def __call__(self, smiles):
        if self.smiles_transform:
            # 改变分子本身
            smiles = self.smiles_transform(smiles)
        smiles = cxsmiles_remove_useless_info(smiles)
        encoded_cxsmiles = cxsmiles_encode(smiles, self.smiles_tokenizer)

        # 切成两部分
        image_smiles, text_smiles = cut_molecule_and_replace_to_r_group(smiles, random.randint(0, self.num_r_groups_in_text))
        image_mol = Chem.MolFromSmiles(image_smiles)
        if self.mol_transform:
            # 改变表示形式，信息不变
            image_mol = self.mol_transform(image_mol)
        real_size = random.randint(int(self.mol_size[0]*0.5), int(self.mol_size[0]*1.3))
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(real_size, real_size)
        drawer_opts = drawer.drawOptions()
        # 字体大小 默认 6/40
        drawer_opts.minFontSize = random.randint(int(6*0.5), int(6*1.5))
        drawer_opts.maxFontSize = random.randint(int(40*0.5), int(40*1.5))
        # # 键长度 默认 无
        # drawer_opts.fixedBondLength = 40
        # 键宽度 默认 2.0
        drawer_opts.bondLineWidth = 2 * random.random() + 1
        # 多键之间间隔 默认 0.15
        drawer_opts.multipleBondOffset = 0.15 * random.random() + 0.075
        # 分子旋转 默认 0.0
        drawer_opts.rotate = 180 * random.random() - 90
        # 图像周围padding 默认0.05
        drawer_opts.padding = 0.3 * random.random()
        # 原子周围空格 默认 0.0
        drawer_opts.additionalAtomLabelPadding = 0.2 * random.random()
        # 显示氢原子 默认 False
        drawer_opts.explicitMethyl = random.random() > 0.5
        # 以80%概率黑白渲染 black and white
        if random.random() < 0.8:
            drawer_opts.useBWAtomPalette()
        drawer.DrawMolecule(image_mol)
        png = bytearray(drawer.GetDrawingText())
        mol_image = Image.open(io.BytesIO(png))
        # 后面再说
        templates = [
            'As an instance, {a} might be "{b}". ',
            'By way of illustration, {a} could be "{b}". ',
            'To give an example, {a} may be "{b}". ',
            'For instance, {a} can be "{b}". ',
            'One possible example is that {a} is "{b}". ',
            'As an example, {a} would be "{b}". ',
            'A concrete example would be "{b}" for {a}. ',
            'As a case in point, {a} should be "{b}". ',
            'A typical example of {a} is "{b}". ',
            'Let\'s say {a} is "{b}" for the sake of argument. ']
        text = ''
        for a, b in text_smiles:
            temp = random.choice(templates)
            b = cxsmiles_remove_useless_info(b)
            text += temp.format(a=a, b=b)
        if self.image_transform:
            mol_image = self.image_transform(mol_image)
        data = {
            'image': mol_image,
            'text': text,
            'smiles': smiles,
            'processed_smiles': encoded_cxsmiles
        }
        if self.return_graph:
            # 改成cx不收敛
            raw_atoms, bonds_graph, bonds_mask = cxsmiles_to_graph(smiles, self.smiles_tokenizer)
            atoms = ''.join(raw_atoms)
            data['raw_atoms'] = ' '.join(raw_atoms)
            data['atoms'] = atoms
            data['bonds_graph'] = torch.from_numpy(bonds_graph)
            data['bonds_mask'] = torch.from_numpy(bonds_mask)
        return data

def collate_fn_with_bond(batch):
    max_len = max([instance['bonds_graph'].shape[0] for instance in batch])  + 1
    out = {}
    for idx, instance in enumerate(batch):
        for k, v in batch[idx].items():
            if k == 'bonds_graph' or k == 'bonds_mask':
                v = F.pad(v, [0, max_len-v.shape[0], 0, max_len-v.shape[0]])
            if k not in out:
                out[k] = [v]
            else:
                out[k].append(v)
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.stack(v)
    return out

if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast
    from transformers import AutoImageProcessor

    mol = Chem.MolFromSmiles('[*]CCC |$R1;;;$|')
    a = mol.GetAtomWithIdx(0)

    n = a.GetPropNames()
    for i in n:
        print(i, a.GetProp(i))

    smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file="config/smiles_tokenizer.json")
    smiles_tokenizer.mask_token = '[MASK]'
    # use [EOS] to pad，这样eos的位置才正确
    smiles_tokenizer.pad_token = '[EOS]'
    smiles_tokenizer.unk_token = '[UNK]'
    smiles_tokenizer.sep_token = '[SEP]'
    smiles_tokenizer.cls_token = '[CLS]'
    smiles_tokenizer.bos_token = '[BOS]'
    smiles_tokenizer.eos_token = '[EOS]'
    smiles_tokenizer.add_tokens(list(RGroupSymbols.keys()), special_tokens=True)

    with open('uspto_data/train.txt', 'r') as fp:
        train_reaction_smarts = fp.readlines()
    train_smiles = set()
    for line in train_reaction_smarts:
        reaction = Chem.AllChem.ReactionFromSmarts(line.strip())
        for mol in reaction.GetReactants():
            smiles = Chem.MolToSmiles(mol)
            train_smiles.add(smiles)
        for mol in reaction.GetProducts():
            smiles = Chem.MolToSmiles(mol)
            train_smiles.add(smiles)
        if len(train_smiles) > 50000:
            break
    train_smiles = list(train_smiles)

    for line in train_smiles:
        raw_smiles = train_smiles[3]
        smiles_without_num = SmilesRemoveNum(1)(raw_smiles)
        canonical_smiles = to_canonical(raw_smiles)

        # atoms, bonds_graph, bonds_mask = smiles_to_graph(raw_smiles)
        # smiles = graph_to_smiles(' '.join(atoms), bonds_graph)
        # smiles_without_num = to_canonical(smiles_without_num)
        # smiles = to_canonical(smiles)
        # if smiles_without_num != smiles:
        #     print('Error1:\n{}\n{}'.format(smiles_without_num, smiles))
        # assert(smiles_without_num == smiles)

        main, sub = cut_molecule_and_replace_to_r_group(smiles, 3)
        cxsmiles = cxsmiles_remove_useless_info(main)
        cxsmiles_for_tokenizer = cxsmiles_encode(cxsmiles, smiles_tokenizer)
        decoded_cxsmiles = cxsmiles_decode(cxsmiles_for_tokenizer, smiles_tokenizer)

        canonical_cxsmiles = to_canonical(cxsmiles)
        canonical_decoded_cxsmiles = to_canonical(decoded_cxsmiles)
        if canonical_cxsmiles != canonical_decoded_cxsmiles:
            print('Error2:\n{}\n{}\n{}\n{}'.format(main, cxsmiles, cxsmiles_for_tokenizer, decoded_cxsmiles))
        assert(canonical_cxsmiles == canonical_decoded_cxsmiles)

        # atoms, bonds_graph, bonds_mask = cxsmiles_to_graph(cxsmiles, smiles_tokenizer)
        # pred_cxsmiles = graph_to_cxsmiles(' '.join(atoms), bonds_graph, smiles_tokenizer)
        # cxsmiles = to_canonical(cxsmiles)
        # pred_cxsmiles = to_canonical(pred_cxsmiles)
        # if cxsmiles != pred_cxsmiles:
        #     print('Error3:\n{}\n{}\n{}'.format(main, cxsmiles, pred_cxsmiles))
        # assert(cxsmiles == pred_cxsmiles)
        # mol = Chem.MolFromSmiles(pred_cxsmiles)

    print('raw Smiles: ', raw_smiles)
    print('Smiles without Num: ', smiles_without_num)
    print('canonical Smiles: ', canonical_smiles)
    # print('atoms: ', atoms)
    # print('bonds graph: ', bonds_graph)
    print('processed Smiles: ', smiles)
    print('cutted Smiles: ', main, sub)
    print('processed CXSmiles: ', cxsmiles)
    print('CXSmiles for Tokenizer: ', cxsmiles_for_tokenizer)
    print('Decoded CXSmiles: ', decoded_cxsmiles)

    smiles_transform = transforms.Compose([SmilesRemoveNum(1), SmilesReplaceWithR(1)])
    image_preprocesser = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window16-256")
    image_transform = ImageTransform(
        transforms.Compose([
            RandomTransform(transforms.GaussianBlur((5, 5)), 0.3),
            transforms.RandomPerspective(0.2, 0.3),
            transforms.RandomAdjustSharpness(0, 0.3),
            RandomNoise(0.3)
            ]), 
            image_preprocesser)

    transform_v1 = TransformV1(
        smiles_tokenizer=smiles_tokenizer,
        smiles_transform=smiles_transform, 
        mol_transform=molecule_abbrevs, 
        image_transform=image_transform, 
        return_graph=False, 
        num_r_groups_in_text=2)

    train_set = Image2SmilesDataset(train_smiles, transform=transform_v1)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False)
    for step, data in enumerate(train_loader):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        if step == 0:
            break

    # transform_v2 = TransformV1(
    #     smiles_tokenizer=smiles_tokenizer,
    #     smiles_transform=smiles_transform, 
    #     mol_transform=molecule_abbrevs, 
    #     image_transform=image_transform, 
    #     return_graph=True)

    # train_set = Image2SmilesDataset(train_smiles, transform=transform_v2)
    # train_loader = DataLoader(train_set, batch_size=16, collate_fn=collate_fn_with_bond)
    # for step, data in enumerate(train_loader):
    #     for k, v in data.items():
    #         if isinstance(v, torch.Tensor):
    #             print(k, v.shape)
    #         else:
    #             print(k, v)
    #     if step == 0:
    #         break