import torch
import torch.nn as nn
import geomloss

def kd_loss(a, b): # a: output features of encoder, b: output features of decoder
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1))) # [4, 256 * 64 * 64]
    return loss

class Separation_loss(nn.Module):
    def __init__(self):
        super(Separation_loss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, reach=None, diameter=10000000, scaling=0.95, truncate=10, cost=None, kernel=None, cluster_scale=None, debias=True, potentials=False, verbose=False, backend='auto')
        self.contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, normal_features, noise_features): # len(3)
        current_batchsize = normal_features[0].shape[0] # 4
        target = -torch.ones(current_batchsize).cuda() # [1., 1., 1., 1.]
        normal_proj1 = normal_features[0] # [4, 256, 64, 64]
        normal_proj2 = normal_features[1] # [4, 512, 32, 32]
        normal_proj3 = normal_features[2] # [4, 1024, 16, 16]
        shuffle_index = torch.randperm(current_batchsize) # [randn(0, 3), randn(0, 3), randn(0, 3), randn(0, 3)]
        shuffle_1 = normal_proj1[shuffle_index] # [4, 256, 64, 64]
        shuffle_2 = normal_proj2[shuffle_index] # [4, 512, 32, 32]
        shuffle_3 = normal_proj3[shuffle_index] # [4, 1024, 16, 16]
        noised_feature1, noised_feature2, noised_feature3 = noise_features # [4, 256, 64, 64], [4, 512, 32, 32], [4, 1024, 16, 16]
        # ot loss
        loss_ssot = self.sinkhorn(torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1), torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1), -1)) +\
                    self.sinkhorn(torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1), -1),  torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1), -1)) +\
                    self.sinkhorn(torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1), -1),  torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1), -1))
        # contrastive loss
        loss_contrast = self.contrast(noised_feature1.view(noised_feature1.shape[0], -1), normal_proj1.view(normal_proj1.shape[0], -1), target=target) +\
                        self.contrast(noised_feature2.view(noised_feature2.shape[0], -1), normal_proj2.view(normal_proj2.shape[0], -1), target=target) +\
                        self.contrast(noised_feature3.view(noised_feature3.shape[0], -1), normal_proj3.view(normal_proj3.shape[0], -1), target=target)
        return loss_ssot + 0.1 * loss_contrast

def norm(data):
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)

def triplet_loss(normal_memory_info, normal_diff_bank, abnormal_memory_info, abnormal_diff_bank):
    margin_loss = nn.TripletMarginLoss(margin=1)
    abnormal_memory_info1, abnormal_memory_info2, abnormal_memory_info3 = abnormal_memory_info['level0'], abnormal_memory_info['level1'], abnormal_memory_info['level2']
    normal_memory_info1, normal_memory_info2, normal_memory_info3 = normal_memory_info['level0'], normal_memory_info['level1'], normal_memory_info['level2']
    # layer 1 of wide-resnet
    _, A_index = torch.topk(abnormal_diff_bank, 3, dim=-1)
    negative_ax1 = torch.gather(abnormal_memory_info1, 1, A_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info1.size(-2), abnormal_memory_info1.size(-1)])).mean(dim=[2, 3])
    _, N_index = torch.topk(normal_diff_bank, 3, dim=-1)
    anchor_nx1 = torch.gather(normal_memory_info1, 1, N_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, normal_memory_info1.size(-2), normal_memory_info1.size(-1)])).mean(dim=[2, 3])
    _, P_index = torch.topk(normal_diff_bank, 3, dim=-1)
    positive_nx1 = torch.gather(abnormal_memory_info1, 1, P_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info1.size(-2), abnormal_memory_info1.size(-1)])).mean(dim=[2, 3])
    # layer 2 of wide-resnet
    negative_ax2 = torch.gather(abnormal_memory_info2, 1, A_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info2.size(-2), abnormal_memory_info2.size(-1)])).mean(dim=[2, 3])
    anchor_nx2 = torch.gather(normal_memory_info2, 1, N_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, normal_memory_info2.size(-2), normal_memory_info2.size(-1)])).mean(dim=[2, 3])
    positive_nx2 = torch.gather(abnormal_memory_info2, 1, P_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info2.size(-2), abnormal_memory_info2.size(-1)])).mean(dim=[2, 3])
    # layer 3 of wide-resnet
    negative_ax3 = torch.gather(abnormal_memory_info3, 1, A_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info3.size(-2), abnormal_memory_info3.size(-1)])).mean(dim=[2, 3])
    anchor_nx3 = torch.gather(normal_memory_info3, 1, N_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, normal_memory_info3.size(-2), normal_memory_info3.size(-1)])).mean(dim=[2, 3])
    positive_nx3 = torch.gather(abnormal_memory_info3, 1, P_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info3.size(-2), abnormal_memory_info3.size(-1)])).mean(dim=[2, 3])
    triplet_margin_loss = margin_loss(norm(anchor_nx1), norm(positive_nx1), norm(negative_ax1)) + margin_loss(norm(anchor_nx2), norm(positive_nx2), norm(negative_ax2)) + margin_loss(norm(anchor_nx3), norm(positive_nx3), norm(negative_ax3))
    return triplet_margin_loss