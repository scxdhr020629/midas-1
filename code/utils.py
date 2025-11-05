import os
import numpy as np
from math import sqrt

from torch_geometric import data as DATA

import os

import torch
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data
import chaos_game_V1  # 确保 chaos_game_V1.py 在您的项目中可用
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp',dataset='',
                 xd=None, xt=None, y=None,z=None, transform=None,
                 pre_transform=None,smile_graph=None,sequence_graph=None,rdkit_fingerprint_dict=None,mogran_fingerprint_dict=None,maccs_fingerprint_dict=None,rdkit_descriptor_dict=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,z,smile_graph,sequence_graph,rdkit_fingerprint_dict,mogran_fingerprint_dict,maccs_fingerprint_dict,rdkit_descriptor_dict)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,z,smile_graph,sequence_graph,rdkit_fingerprint_dict=None,mogran_fingerprint_dict=None,maccs_fingerprint_dict=None,rdkit_descriptor_dict=None):
        count=0
        print(len(xd),len(xt),'====',len(y))
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            seqdrug=z[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            #print(c_size,features,edge_index)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            if len(edge_index) == 0:
                count=count+1
                print(f'No edges for graph {i + 1}, skipping...',smiles)

                continue
            # rdkit_fingerprint = rdkit_fingerprint_dict[smiles]
            # mogran_fingerprint = mogran_fingerprint_dict[smiles]
            # maccs_fingerprint = maccs_fingerprint_dict[smiles]
            # rdkit_descriptor = rdkit_descriptor_dict[smiles]
            # 获取四个药物特征并进行非空检测
            try:
                # 检查字典是否为 None
                if None in [rdkit_fingerprint_dict, mogran_fingerprint_dict, maccs_fingerprint_dict,
                            rdkit_descriptor_dict]:
                    raise ValueError(f"One or more feature dictionaries is None at index {i}")

                # 获取特征
                rdkit_fingerprint = rdkit_fingerprint_dict[smiles]
                mogran_fingerprint = mogran_fingerprint_dict[smiles]
                maccs_fingerprint = maccs_fingerprint_dict[smiles]
                rdkit_descriptor = rdkit_descriptor_dict[smiles]

                # 检查特征是否为 None
                if None in [rdkit_fingerprint, mogran_fingerprint, maccs_fingerprint, rdkit_descriptor]:
                    raise ValueError(f"One or more features is None for SMILES: {smiles}")

            except (KeyError, ValueError) as e:
                print(f"[Error] Index {i}, SMILES: {smiles}, Error: {str(e)}")
                count += 1
                continue
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.LongTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # Add seqdrug as an attribute
            # GCNData.__setitem__('seqdrug', torch.FloatTensor([seqdrug]))
            # test long
            GCNData.__setitem__('seqdrug', torch.LongTensor([seqdrug]))
            # 存rdkit_fingerprint四个指纹

            # 将四个特征转换为 Tensor 并存储
            GCNData.__setitem__('rdkit_fingerprint', torch.FloatTensor(rdkit_fingerprint))
            GCNData.__setitem__('morgan_fingerprint', torch.FloatTensor(mogran_fingerprint))
            GCNData.__setitem__('maccs_fingerprint', torch.FloatTensor(maccs_fingerprint))
            GCNData.__setitem__('rdkit_descriptor', torch.FloatTensor(rdkit_descriptor))
            # ==========================================



            target_array = xt[i]
            key_target = tuple(target_array)
            target_matrix = sequence_graph.get(key_target)
            if target_matrix is not None:
                # 转换为Tensor，并增加一个“通道”维度以适配2D CNN (变为 [1, H, W])
                target_matrix_tensor = torch.FloatTensor(target_matrix).unsqueeze(0)
                # 将矩阵存入数据对象
                GCNData.__setitem__('target_matrix', target_matrix_tensor)
            else:
                print(f"Warning: Target sequence key not found in sequence_graph for item {i + 1}. Skipping matrix.")
                # 如果找不到，可以跳过，或存入一个全零矩阵
                # 这里我们选择不存入该属性，后续模型代码需要做相应处理
                # GCNData.__setitem__('target_matrix', torch.zeros(1, H, W)) # H, W 是你的矩阵维度
            ## --- 新增代码结束 ---




            # append graph, label and target sequence to data list
            data_list.append(GCNData)
        # print("去除不规则数量", count, "总数量为", data_len)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

