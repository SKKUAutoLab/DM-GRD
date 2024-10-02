import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from metrics.pro_curve_util import compute_pro
from metrics.generic_util import trapezoid
from scipy.ndimage import gaussian_filter
import torch.nn as nn
import geomloss
import matplotlib.pyplot as plt

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def loss_function(a, b): # a: output features of encoder, b: output features of decoder
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1))) # [4, 256 * 64 * 64]
    return loss

def get_concatenated_features(features1, features2):
    cfeatures = []
    for layer_id in range(len(features1)):
        fi = features1[layer_id]  # (B, dim, h, w)
        pi = features2[layer_id]  # (B, dim, h, w)
        ci = torch.cat([fi, pi], dim=1)
        cfeatures.append(ci)
    return cfeatures

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
        # shuffling samples order for computing pair-wise loss_ssot in batch-mode (for efficient computation)
        shuffle_index = torch.randperm(current_batchsize) # [randn(0, 3), randn(0, 3), randn(0, 3), randn(0, 3)]
        shuffle_1 = normal_proj1[shuffle_index] # [4, 256, 64, 64]
        shuffle_2 = normal_proj2[shuffle_index] # [4, 512, 32, 32]
        shuffle_3 = normal_proj3[shuffle_index] # [4, 1024, 16, 16]
        noised_feature1, noised_feature2, noised_feature3 = noise_features # [4, 256, 64, 64], [4, 512, 32, 32], [4, 1024, 16, 16]
        # loss
        loss_ssot = self.sinkhorn(torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1), torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1), -1)) +\
                    self.sinkhorn(torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1), -1),  torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1), -1)) +\
                    self.sinkhorn(torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1), -1),  torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1), -1))
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

    _, A_index = torch.topk(abnormal_diff_bank, 3, dim=-1)
    negative_ax1 = torch.gather(abnormal_memory_info1, 1, A_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info1.size(-2), abnormal_memory_info1.size(-1)])).mean(dim=[2, 3])
    _, N_index = torch.topk(normal_diff_bank, 3, dim=-1)
    anchor_nx1 = torch.gather(normal_memory_info1, 1, N_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, normal_memory_info1.size(-2), normal_memory_info1.size(-1)])).mean(dim=[2, 3])
    _, P_index = torch.topk(normal_diff_bank, 3, dim=-1)
    positive_nx1 = torch.gather(abnormal_memory_info1, 1, P_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info1.size(-2), abnormal_memory_info1.size(-1)])).mean(dim=[2, 3])

    negative_ax2 = torch.gather(abnormal_memory_info2, 1, A_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info2.size(-2), abnormal_memory_info2.size(-1)])).mean(dim=[2, 3])
    anchor_nx2 = torch.gather(normal_memory_info2, 1, N_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, normal_memory_info2.size(-2), normal_memory_info2.size(-1)])).mean(dim=[2, 3])
    positive_nx2 = torch.gather(abnormal_memory_info2, 1, P_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info2.size(-2), abnormal_memory_info2.size(-1)])).mean(dim=[2, 3])

    negative_ax3 = torch.gather(abnormal_memory_info3, 1, A_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info3.size(-2), abnormal_memory_info3.size(-1)])).mean(dim=[2, 3])
    anchor_nx3 = torch.gather(normal_memory_info3, 1, N_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, normal_memory_info3.size(-2), normal_memory_info3.size(-1)])).mean(dim=[2, 3])
    positive_nx3 = torch.gather(abnormal_memory_info3, 1, P_index.unsqueeze(2).unsqueeze(3).expand([-1, -1, abnormal_memory_info3.size(-2), abnormal_memory_info3.size(-1)])).mean(dim=[2, 3])

    triplet_margin_loss = margin_loss(norm(anchor_nx1), norm(positive_nx1), norm(negative_ax1)) + margin_loss(norm(anchor_nx2), norm(positive_nx2), norm(negative_ax2)) +\
                          margin_loss(norm(anchor_nx3), norm(positive_nx3), norm(negative_ax3))
    return triplet_margin_loss

def visualize_loss(epoch, loss, args):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epoch, loss, label='Total loss', color='purple')
    # ax.set_title('Visualization of Properties')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    os.makedirs('vis_results/' + args.target, exist_ok=True)
    plt.savefig('vis_results/' + args.target + '/vis_loss.png', dpi=300)

def visualize_result(epoch, img_auc, pixel_auc, pixel_pro, args):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epoch, img_auc, label='Image-level AUC', color='blue')
    ax.plot(epoch, pixel_auc, label='Pixel-level AUC', color='green')
    ax.plot(epoch, pixel_pro, label='Pixel-level PRO', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.legend()
    plt.savefig('vis_results/' + args.target + '/vis_result.png', dpi=300)

def training(teacher, bn, student, msff, memory_bank_normal, memory_bank_abnormal, trainloader, validloader, optimizer, scheduler, num_training_steps=1000, log_interval=1, eval_interval=1, savedir=None, args=None):
    losses_m = AverageMeter()
    # loss
    separation_criterion = Separation_loss()
    # model
    teacher.eval()
    student.train()
    bn.train()
    msff.train()
    optimizer.zero_grad()
    # train
    best_score = 0
    best_img_auc = 0
    best_pixel_auc = 0
    best_pixel_pro = 0
    step = 0
    train_mode = True
    # visualize loss + auc
    vis_img_auc = []
    vis_pixel_auc = []
    vis_pixel_pro = []
    vis_loss = []
    vis_epoch = []
    while train_mode:
        for img in trainloader: # [8, 3, 256, 256], [8, 256, 256], [8]
            img = img.cuda()
            inputs = teacher(img)
            normal_memory_info, normal_diff_bank, concat_features_normal = memory_bank_normal.select(features1=inputs)
            msff_outputs_normal = msff(features=concat_features_normal)
            abnormal_memory_info, abnormal_diff_bank, concat_features_abnormal = memory_bank_abnormal.select(features1=inputs)
            msff_outputs_abnormal = msff(features=concat_features_abnormal)
            outputs = student(bn(msff_outputs_normal))
            # loss = loss_function(inputs, outputs) + 0.2 * separation_criterion(msff_outputs_normal, msff_outputs_abnormal)
            loss = loss_function(inputs, outputs) + 0.2 * separation_criterion(msff_outputs_normal, msff_outputs_abnormal) + 0.1 * triplet_loss(normal_memory_info, normal_diff_bank, abnormal_memory_info, abnormal_diff_bank)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())
            if (step + 1) % log_interval == 0 or step == 0:
                print('Epoch: [{}/{}], Loss: {:.4f}'.format(step + 1, num_training_steps, losses_m.avg))
            if ((step + 1) % eval_interval == 0 and step != 0) or (step + 1) == num_training_steps:
                eval_metrics = evaluate(teacher=teacher, bn=bn, student=student, msff=msff, memory_bank_normal=memory_bank_normal, dataloader=validloader, args=args)
                vis_epoch.append(step + 1)
                vis_img_auc.append(eval_metrics['AUROC-image'])
                vis_pixel_auc.append(eval_metrics['AUROC-pixel'])
                vis_pixel_pro.append(eval_metrics['AUPRO-pixel'])
                vis_loss.append(losses_m.avg)
                visualize_loss(vis_epoch, vis_loss, args)
                visualize_result(vis_epoch, vis_img_auc, vis_pixel_auc, vis_pixel_pro, args)
                teacher.eval()
                student.train()
                bn.train()
                msff.train()
                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
                # save best model
                if best_score < np.mean(list(eval_metrics.values())):
                    state = {'best_step': step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'), 'w'), indent='\t')
                    torch.save({'msff': msff.state_dict(), 'student': student.state_dict(), 'bn': bn.state_dict()}, os.path.join(savedir, f'best_model.pt'))
                    best_score = np.mean(list(eval_metrics.values()))
                    best_img_auc = eval_metrics['AUROC-image']
                    best_pixel_auc = eval_metrics['AUROC-pixel']
                    best_pixel_pro = eval_metrics['AUPRO-pixel']
            if scheduler:
                scheduler.step()
            step += 1
            if step == num_training_steps:
                train_mode = False
                break
    # save latest score
    state = {'latest_step': step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'), 'w'), indent='\t')
    return best_img_auc, best_pixel_auc, best_pixel_pro

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def evaluate(teacher, bn, student, msff, memory_bank_normal, dataloader, args):
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []
    teacher.eval()
    bn.eval()
    student.eval()
    msff.eval()
    with torch.no_grad():
        for idx, (img, masks, targets) in enumerate(dataloader): # [8, 3, 256, 256], [8, 256, 256], [8]
            img, masks, targets = img.cuda(), masks.cuda(), targets.cuda()
            inputs = teacher(img)
            _, _, concat_features_normal = memory_bank_normal.select(features1=inputs)
            msff_outputs_normal = msff(features=concat_features_normal)
            outputs = student(bn(msff_outputs_normal))
            scores, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            scores = gaussian_filter(scores, sigma=4)
            scores = torch.from_numpy(scores).unsqueeze(0).cuda()
            anomaly_score_i = torch.topk(torch.flatten(scores, start_dim=1), 100)[0].mean(dim=1) # [8]
            image_targets.extend(targets.cpu().tolist())
            image_masks.extend(masks.cpu().numpy())
            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(scores.cpu().numpy())
    image_masks = np.array(image_masks) # [83, 256, 256]
    anomaly_map = np.array(anomaly_map) # [83, 256, 256]
    auroc_image = roc_auc_score(image_targets, anomaly_score) # image-level auc
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1)) # pixel-level auc
    all_fprs, all_pros = compute_pro(anomaly_maps=anomaly_map, ground_truth_maps=image_masks)
    aupro = trapezoid(all_fprs, all_pros) # pixel-level pro
    metrics = {'AUROC-image': auroc_image, 'AUROC-pixel': auroc_pixel, 'AUPRO-pixel': aupro}
    print('Class: {}, Image-level AUC: {:.4f}, Pixel-level AUC: {:.4f}, Pixel-level PRO: {:.4f}'.format(args.target, auroc_image, auroc_pixel, aupro))
    return metrics
