import torch
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from data.dataset import MVTecDataset_train, MVTecDataset_test, get_data_transforms
import numpy as np
import random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from metrics.pro_curve_util import compute_pro
from metrics.generic_util import trapezoid
from msff import MSFF
from memory_module import MemoryBank

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size]) # [256, 256]
    else:
        anomaly_map = np.zeros([out_size, out_size]) # [256, 256]
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i] # [1, 256, 64, 64]
        ft = ft_list[i] # [1, 256, 64, 64]
        a_map = 1 - F.cosine_similarity(fs, ft) # [1, 64, 64]
        a_map = torch.unsqueeze(a_map, dim=1) # [1, 1, 64, 64]
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True) # [1, 1, 256, 256]
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy() # [256, 256]
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
    count = 0
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
            count += 1
    image_masks = np.array(image_masks) # [83, 256, 256]
    anomaly_map = np.array(anomaly_map) # [83, 256, 256]
    auroc_image = roc_auc_score(image_targets, anomaly_score) # image-level auc
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1)) # pixel-level auc
    all_fprs, all_pros = compute_pro(anomaly_maps=anomaly_map, ground_truth_maps=image_masks)
    aupro = trapezoid(all_fprs, all_pros) # pixel-level pro
    print('Class: {}, Image-level AUC: {:.4f}, Pixel-level AUC: {:.4f}, Pixel-level PRO: {:.4f}'.format(args.target, auroc_image, auroc_pixel, aupro))
    return auroc_image, auroc_pixel, aupro

def inference(_class_, args):
    setup_seed(args.seed)
    if args.type_dataset == 'mvtec':
        train_path = 'datasets/mvtec/' + _class_ + '/train'
        test_path = 'datasets/mvtec/' + _class_
    elif args.type_dataset == 'btad':
        train_path = 'datasets/btad/' + _class_ + '/train'
        test_path = 'datasets/btad/' + _class_
    elif args.type_dataset == 'visa':
        train_path = 'datasets/visa/' + _class_ + '/train'
        test_path = 'datasets/visa/' + _class_
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    data_transform, gt_transform = get_data_transforms(args.image_size)
    # test loader
    testset = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform, type_dataset=args.type_dataset)
    testloader = DataLoader(dataset=testset, shuffle=False, batch_size=1)
    memoryset_normal = MVTecDataset_train(root=train_path, transform=data_transform, type_dataset=args.type_dataset, dtd_paths=args.texture_source_dir, to_memory_normal=True, to_memory_abnormal=False)
    # model
    teacher, bn = wide_resnet50_2(pretrained=True)
    teacher, bn = teacher.cuda(), bn.cuda()
    teacher.eval()
    student = de_wide_resnet50_2(pretrained=False).cuda()
    memory_bank_normal = MemoryBank(normal_dataset=memoryset_normal, nb_memory_sample=args.nb_memory_sample, is_normal=True)
    memory_bank_normal.update(teacher=teacher)
    msff = MSFF().cuda()
    checkpoint_class = args.checkpoint_folder + '/' + _class_ + '/' + 'best_model.pt'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    bn.load_state_dict(ckp['bn'])
    student.load_state_dict(ckp['student'])
    msff.load_state_dict(ckp['msff'])
    # eval
    img_auc, pixel_auc, pixel_pro = evaluate(teacher=teacher, bn=bn, student=student, msff=msff, memory_bank_normal=memory_bank_normal, dataloader=testloader, args=args)
    return img_auc, pixel_auc, pixel_pro

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type_dataset', choices=['mvtec', 'btad', 'visa'], default='mvtec', type=str)
    parser.add_argument('--checkpoint_folder', default='saved_mvtec', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--nb_memory_sample', default=30, type=int)
    parser.add_argument('--texture_source_dir', default='datasets/dtd/images', type=str)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    print('Evaluating dataset:', args.type_dataset)
    if args.type_dataset == 'mvtec':
        all_classes = ['carpet', 'leather', 'grid', 'tile', 'wood', 'bottle', 'hazelnut', 'cable', 'capsule', 'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper'] # 15 objects
    elif args.type_dataset == 'btad':
        all_classes = ['01', '02', '03'] # 3 objects
    elif args.type_dataset == 'visa':
        all_classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'] # 12 objects
    else:
        print('This dataset does not exist')
        raise NotImplementedError
    list_img_auc, list_pixel_auc, list_pixel_pro = [], [], []
    for cls in all_classes:
        args.target = cls
        img_auc, pixel_auc, pixel_pro = inference(cls, args)
        list_img_auc.append(img_auc)
        list_pixel_auc.append(pixel_auc)
        list_pixel_pro.append(pixel_pro)
    print('Image-level AUC:', np.round(np.mean(list_img_auc), 4))
    print('Pixel-level AUC:', np.round(np.mean(list_pixel_auc), 4))
    print('Pixel-level PRO:', np.round(np.mean(list_pixel_pro), 4))