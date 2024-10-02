import json
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from metrics.pro_curve_util import compute_pro
from metrics.generic_util import trapezoid
from scipy.ndimage import gaussian_filter
from losses import Separation_loss, kd_loss, triplet_loss
from utils import AverageMeter
from test import cal_anomaly_map

def training(teacher, bn, student, msff, memory_bank_normal, memory_bank_abnormal, trainloader, validloader, optimizer, scheduler, num_training_steps=1000, log_interval=1, eval_interval=1, savedir=None, args=None):
    losses_m = AverageMeter()
    loss_kd_m = AverageMeter()
    loss_seperation_m = AverageMeter()
    loss_triplet_m = AverageMeter()
    # loss
    separation_criterion = Separation_loss()
    # model
    teacher.eval()
    student.train()
    bn.train()
    msff.train()
    optimizer.zero_grad()
    # metrics
    best_score = 0
    best_img_auc = 0
    best_pixel_auc = 0
    best_pixel_pro = 0
    step = 0
    # train
    train_mode = True
    while train_mode:
        for img in trainloader: # [8, 3, 256, 256], [8, 256, 256], [8]
            img = img.cuda()
            inputs = teacher(img)
            normal_memory_info, normal_diff_bank, concat_features_normal = memory_bank_normal.select(features1=inputs)
            msff_outputs_normal = msff(features=concat_features_normal)
            abnormal_memory_info, abnormal_diff_bank, concat_features_abnormal = memory_bank_abnormal.select(features1=inputs)
            msff_outputs_abnormal = msff(features=concat_features_abnormal)
            outputs = student(bn(msff_outputs_normal))
            loss_kd = kd_loss(inputs, outputs)
            loss_sep = separation_criterion(msff_outputs_normal, msff_outputs_abnormal)
            loss_triplet = triplet_loss(normal_memory_info, normal_diff_bank, abnormal_memory_info, abnormal_diff_bank)
            loss = loss_kd + 0.2 * loss_sep + 0.1 * loss_triplet
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())
            loss_kd_m.update(loss_kd.item())
            loss_seperation_m.update(loss_sep.item())
            loss_triplet_m.update(loss_triplet.item())
            if (step + 1) % log_interval == 0 or step == 0:
                print('Epoch: [{}/{}], Class: {}, Total Loss: {:.4f}, KD loss: {:.4f}, Seperation loss: {:.4f}, Triplet loss: {:.4f}'.format(step + 1, num_training_steps, args.target, losses_m.avg, loss_kd_m.avg, loss_seperation_m.avg, loss_triplet_m.avg))
            if ((step + 1) % eval_interval == 0 and step != 0) or (step + 1) == num_training_steps:
                eval_metrics = evaluate(epoch=step + 1, teacher=teacher, bn=bn, student=student, msff=msff, memory_bank_normal=memory_bank_normal, dataloader=validloader, args=args)
                teacher.eval()
                student.train()
                bn.train()
                msff.train()
                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
                # save checkpoint
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
    return best_img_auc, best_pixel_auc, best_pixel_pro

def evaluate(epoch, teacher, bn, student, msff, memory_bank_normal, dataloader, args):
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
            scores = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
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
    print('Epoch: [{}/{}], Class: {}, Image-level AUC: {:.4f}, Pixel-level AUC: {:.4f}, Pixel-level PRO: {:.4f}'.format(epoch, args.num_training_steps, args.target, auroc_image, auroc_pixel, aupro))
    return metrics