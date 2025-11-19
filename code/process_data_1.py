import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict

from openpyxl.descriptors import Descriptor
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem, rdMolDescriptors, Descriptors
import networkx as nx
from utils import *
import chaos_game_V1

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        # print(line[0][0])
        # print(i,"========", ch)
        X[i] = smi_ch_ind[ch]
    return X

def seq_cat(prot,max_seq_len):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ">": 65, "<": 66}

CHARISOSMILEN = 66



#all_prots 列表中存储了所有数据集中出现的蛋白质序列，这些蛋白质序列将被用于后续的数据处理和模型训练。
all_prots = []
fpath='data/'
# 读取整个 Excel 文件
drugs = pd.read_excel('data/drug_name_with_smiles.xlsx')

rna = pd.read_excel('data/miRNA_sequence_with_seq.xlsx')
ligands = drugs['SMILES']
proteins = rna['Sequence']

# ass=pd.read_excel('data/miRNA_drug_matrix.xlsx', index_col=0)   # 关系矩阵导入
# Positive = json.load(open("data/Positive.txt"))
# Negetive = json.load(open("data/Negative.txt"))
# ass=pd.read_excel('data/sensitive_matrix.xlsx', index_col=0)
#
# Positive = json.load(open("data/sensitive_Positive_fork2.txt"))
# Negetive = json.load(open("data/sensitive_Negative_fork2.txt"))
ass=pd.read_excel('data/resistant_matrix.xlsx', index_col=0)

Positive = json.load(open("data/resistant_Positive.txt"))
Negetive = json.load(open("data/resistant_Negative.txt"))
Potrain_fold=[[] for i in range(5)]
Netrain_fold=[[] for i in range(5)]
Povalid_fold=[[] for i in range(5)]
Nevalid_fold=[[] for i in range(5)]
Po_subset = [[] for i in range(5)]
Ne_subset=[[] for i in range(5)]
for i in  range(5):
    Po_subset[i] = [ee for ee in Positive[i]]
    Ne_subset[i] = [ee for ee in Negetive[i]]
for i in range(5):
    for j in range(5):
        if i == j:
            continue
        Potrain_fold[i] += Po_subset[j]
        Netrain_fold[i] += Ne_subset[j]
    Povalid_fold[i] = Po_subset[i]
    Nevalid_fold[i] = Ne_subset[i]

drugs = []
prots = []
for d in ligands.keys():
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
    drugs.append(lg)
for t in proteins.keys():
    prots.append(proteins[t])

affinity = np.asarray(ass)

opts = ['train', 'test']

for opt in opts:
    po_rows, po_cols = np.where(ass.values == 1)
    ne_rows, ne_cols = np.where(ass.values != 1)
    for i in range(5):
        if opt == 'train':
            rows = np.concatenate((po_rows[Potrain_fold[i]], ne_rows[Netrain_fold[i]]))
            cols = np.concatenate((po_cols[Potrain_fold[i]], ne_cols[Netrain_fold[i]]))
        elif opt == 'test':
            rows = np.concatenate((po_rows[Povalid_fold[i]], ne_rows[Nevalid_fold[i]]))
            cols = np.concatenate((po_cols[Povalid_fold[i]], ne_cols[Nevalid_fold[i]]))

        with open('data/processed/'+ opt +str(i)+ '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[cols[pair_ind]]]
                ls += [prots[rows[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')


all_prots += list(set(prots))
seq_voc = "ACGU"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)

compound_iso_smiles = []
rna_sequences = []
# 换一种rna sequence 的读取方法
# rna_sequences_1 = []
# df = pd.read_excel('data/miRNA_sequences.xlsx')
# rna_sequences_1 += list(df['Sequence'])
# rna_sequences_1 = set(rna_sequences_1)
# sequence1_graph = {}
# for seq in rna_sequences_1:
#     temp = chaos_game.generate_cgr_matrix(seq)
#     sequence1_graph[seq] = temp


opts = ['train', 'test']

for opt in opts:
    df = pd.read_csv('data/processed/' + opt +str(i)+ '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
    rna_sequences += list(df['target_sequence'])
compound_iso_smiles = set(compound_iso_smiles)
rna_sequences = set(rna_sequences)
smile_graph = {}
sequence_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

for seq in rna_sequences:
    one_hot_seq = seq_cat(seq, 24)
    key_seq = tuple(one_hot_seq)  # 将numpy数组转换为元组以用作字典键
    temp = chaos_game_V1.generate_cgr_matrix(seq)
    sequence_graph[key_seq] = temp

rdkit_fingerprint_dict = {}
mogran_fingerprint_dict = {}
maccs_fingerprint_dict = {}
rdkit_descriptor_dict = {}
print("hello")
# 写分子图逻辑
for smile in compound_iso_smiles:
    mol = Chem.MolFromSmiles(smile)
    # rdkit fingerprint 转换为整数列表
    fp_1 = Chem.RDKFingerprint(mol, fpSize=136)
    bit_string_1 = fp_1.ToBitString()
    bit_list_1 = [int(bit) for bit in bit_string_1]
    # Morgan fingerprint
    fp_2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    bit_string_2 = fp_2.ToBitString()
    bit_list_2 = [int(bit) for bit in bit_string_2]
    # MACCS fingerprint
    fp_3 = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    bit_string_3 = fp_3.ToBitString()
    bit_list_3 = [int(bit) for bit in bit_string_3]
    bit_list_3 = bit_list_3[1:]
    # rdkit descriptor
    descriptor_list = Descriptors._descList
    descriptor_vector = [desc_func(mol) for desc_name, desc_func in descriptor_list]
    # print(smile)
    lg = Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](CCCC(C)(C)O)[C@@]1([H])CC[C@@]2([H])\\C(CCC[C@]12C)=C\\C=C1\\C[C@@H](O)C[C@H](O)C1=C"), isomericSmiles=True)
    # if smile == lg:
    #     print('rdkit fingerprint:', bit_list_1)
    #     print('Morgan fingerprint:', bit_list_2)
    #     print('MACCS fingerprint:', bit_list_3)
    #     print('rdkit descriptor:', descriptor_vector)
    rdkit_fingerprint_dict[smile] = bit_list_1
    mogran_fingerprint_dict[smile] = bit_list_2
    maccs_fingerprint_dict[smile] = bit_list_3
    rdkit_descriptor_dict[smile] = descriptor_vector



# 数据处理转换成需要的格式内容
for i in range(5):
    processed_data_file_train = 'data/processed/'  + '_train'+str(i)+'.pt'
    processed_data_file_test = 'data/processed/'+ '_test'+str(i)+'.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/processed/train' + str(i)+ '.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t,24) for t in train_prots]
        train_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in train_drugs]
        train_drugs, train_prots, train_Y,train_seqdrugs = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y),np.asarray(train_sdrugs)
        df = pd.read_csv('data/processed/test'+str(i)+ '.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t,24) for t in test_prots]
        test_sdrugs=[label_smiles(t,CHARISOSMISET,100) for t in test_drugs]
        test_drugs, test_prots, test_Y,test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y),np.asarray(test_sdrugs)


        print('preparing NO:', i,'train.pt in pytorch format!')

        train_data = TestbedDataset(root='data/',dataset='train'+str(i), xd=train_drugs, xt=train_prots, y=train_Y, z=train_seqdrugs,
                                    smile_graph=smile_graph,sequence_graph=sequence_graph,rdkit_fingerprint_dict=rdkit_fingerprint_dict,mogran_fingerprint_dict=mogran_fingerprint_dict,maccs_fingerprint_dict=maccs_fingerprint_dict,rdkit_descriptor_dict=rdkit_descriptor_dict)
        print('preparing ',   '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data/',dataset='test'+str(i),  xd=test_drugs, xt=test_prots, y=test_Y, z=test_seqdrugs,
                                   smile_graph=smile_graph,sequence_graph=sequence_graph,rdkit_fingerprint_dict=rdkit_fingerprint_dict,mogran_fingerprint_dict=mogran_fingerprint_dict,maccs_fingerprint_dict=maccs_fingerprint_dict,rdkit_descriptor_dict=rdkit_descriptor_dict)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
