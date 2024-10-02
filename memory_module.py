import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample=30, is_normal=True):
        self.is_normal = is_normal
        self.normal_memory_information = {}
        self.abnormal_memory_information = {}
        self.normal_dataset = normal_dataset
        self.nb_memory_sample = nb_memory_sample
        self.softmax = nn.Softmax(dim=1)
        self.update_normal_memory_information = {}
        self.update_abnormal_memory_information = {}

    def update(self, teacher):
        teacher.eval()
        samples_idx = np.arange(len(self.normal_dataset)) # [280]
        np.random.shuffle(samples_idx)
        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                input_normal = self.normal_dataset[samples_idx[i]] # [3, 256, 256]
                input_normal = input_normal.cuda()
                features = teacher(input_normal.unsqueeze(0)) # [1, 64, 64, 64], [1, 128, 32, 32], [1, 256, 16, 16]
                # extract features for normal and abnormal memory information
                if self.is_normal:
                    mem_info = self.normal_memory_information
                else:
                    mem_info = self.abnormal_memory_information
                for i, features_l in enumerate(features):
                    if f'level{i}' not in mem_info.keys():
                        mem_info[f'level{i}'] = features_l.mean(dim=[2, 3])
                    else:
                        mem_info[f'level{i}'] = torch.cat([mem_info[f'level{i}'], features_l.mean(dim=[2, 3])], dim=0) # [30, 64, 64, 64], [30, 128, 32, 32], [30, 256, 16, 16]
                # extract features for updatable normal and abnormal memory information
                if self.is_normal:
                    update_mem_info = self.update_normal_memory_information
                else:
                    update_mem_info = self.update_abnormal_memory_information
                for i, features_l in enumerate(features):
                    if f'level{i}' not in update_mem_info.keys():
                        update_mem_info[f'level{i}'] = features_l
                    else:
                        update_mem_info[f'level{i}'] = torch.cat([update_mem_info[f'level{i}'], features_l], dim=0)
                        
    def _calc_diff(self, features1): # [8, 64, 64, 64], [8, 128, 32, 32], [8, 256, 16, 16]
        features = [features1[i].clone() for i in range(len(features1))]
        diff_bank = []
        if self.is_normal:
            mem_info = self.normal_memory_information
        else:
            mem_info = self.abnormal_memory_information
        for l, level in enumerate(mem_info.keys()):
            b, c, h, w = features[l].size()
            features[l] = features[l].view(features[l].size(0), features[l].size(1), -1).permute(0, 2, 1)
            input_norm = mem_info[level].unsqueeze(0).expand(features[l].size(0), -1, -1)
            input_norm = F.normalize(input_norm, p=2, dim=2)
            target_norm = F.normalize(features[l], p=2, dim=2)
            cos_sim = torch.matmul(input_norm.unsqueeze(2), target_norm.transpose(1, 2).unsqueeze(1)).squeeze(2)
            sim_vec = self.softmax(cos_sim)
            Fnorm = torch.matmul(sim_vec.permute(0, 2, 1), mem_info[level])
            Fr = Fnorm.permute(0, 2, 1).view(b, c, h, w)
            top_Fr = torch.topk(Fr, k=self.nb_memory_sample // 3, dim=1)[0].mean(dim=[2, 3])
            diff_bank.append(top_Fr)
        diff_bank = torch.cat(diff_bank, dim=1)
        return diff_bank

    def select(self, features1): # [8, 64, 64, 64], [8, 128, 32, 32], [8, 256, 16, 16]
        features = [features1[i].clone() for i in range(len(features1))]
        # calculate difference between features and normal features of memory bank
        diff_bank = self._calc_diff(features1=features)
        if self.is_normal:
            mem_info = self.update_normal_memory_information
        else:
            mem_info = self.update_abnormal_memory_information
        for l, level in enumerate(mem_info.keys()):
            selected_features = torch.index_select(mem_info[level], dim=0, index=diff_bank.argmin(dim=1)) # [8, 64, 64, 64], [8, 128, 32, 32], [8, 256, 16, 16]
            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)
        if self.is_normal:
            return self.update_normal_memory_information, diff_bank, features
        else:
            return self.update_abnormal_memory_information, diff_bank, features