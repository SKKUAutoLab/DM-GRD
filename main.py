import os
import torch
from train import training
from utils import CosineAnnealingWarmupRestarts
from argparse import ArgumentParser
import numpy as np
import random
from torch.utils.data import DataLoader
from data.dataset import MVTecDataset_train, MVTecDataset_test, get_data_transforms
from models.memory_module import MemoryBank
from models.resnet import wide_resnet50_2
from models.de_resnet import de_wide_resnet50_2
from models.msff import MSFF

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    setup_seed(args.seed)
    print('Training class:', args.target)
    savedir = os.path.join(args.save_dir, args.target)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # train and test loader
    data_transform, gt_transform = get_data_transforms(args.imagesize)
    if args.type_dataset == 'mvtec':
        train_path = 'datasets/mvtec/' + args.target + '/train'
        test_path = 'datasets/mvtec/' + args.target
    elif args.type_dataset == 'btad':
        train_path = 'datasets/btad/' + args.target + '/train'
        test_path = 'datasets/btad/' + args.target
    elif args.type_dataset == 'visa':
        train_path = 'datasets/visa/' + args.target + '/train'
        test_path = 'datasets/visa/' + args.target
    else:
        print('This datset does not exist')
        raise NotImplementedError
    trainset = MVTecDataset_train(root=train_path, transform=data_transform, type_dataset=args.type_dataset, dtd_paths=args.texture_source_dir, to_memory_normal=False, to_memory_abnormal=False)
    testset = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform, type_dataset=args.type_dataset)
    memoryset_normal = MVTecDataset_train(root=train_path, transform=data_transform, type_dataset=args.type_dataset, dtd_paths=args.texture_source_dir, to_memory_normal=True, to_memory_abnormal=False)
    memoryset_abnormal = MVTecDataset_train(root=train_path, transform=data_transform, type_dataset=args.type_dataset, dtd_paths=args.texture_source_dir, to_memory_normal=False, to_memory_abnormal=True)
    trainloader = DataLoader(dataset=trainset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    testloader = DataLoader(dataset=testset, shuffle=False, batch_size=1, num_workers=args.num_workers)
    # model
    teacher, bn = wide_resnet50_2(pretrained=True)
    teacher, bn = teacher.cuda(), bn.cuda()
    teacher.eval()
    student = de_wide_resnet50_2(pretrained=False).cuda()
    memory_bank_normal = MemoryBank(normal_dataset=memoryset_normal, nb_memory_sample=args.nb_memory_sample, is_normal=True)
    memory_bank_abnormal = MemoryBank(normal_dataset=memoryset_abnormal, nb_memory_sample=args.nb_memory_sample, is_normal=False)
    memory_bank_normal.update(teacher=teacher)
    memory_bank_abnormal.update(teacher=teacher)
    msff = MSFF().cuda()
    print('Total parameters of the model:', count_parameters(teacher) + count_parameters(student) + count_parameters(msff) + count_parameters(bn))
    # optimizer
    optimizer = torch.optim.Adam(list(student.parameters()) + list(bn.parameters()) + list(msff.parameters()), lr=args.lr, betas=(0.5, 0.999))
    # scheduler
    if args.use_scheduler:
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.num_training_steps, max_lr=args.lr, min_lr=args.min_lr, warmup_steps=int(args.num_training_steps * args.warmup_ratio))
    else:
        scheduler = None
    best_img_auc, best_pixel_auc, best_pixel_pro = training(teacher=teacher, bn=bn, student=student, msff=msff, memory_bank_normal=memory_bank_normal, memory_bank_abnormal=memory_bank_abnormal, num_training_steps=args.num_training_steps, trainloader=trainloader,
                                                            validloader=testloader, optimizer=optimizer, scheduler=scheduler, log_interval=args.log_interval, eval_interval=args.eval_interval, savedir=savedir, args=args)
    return best_img_auc, best_pixel_auc, best_pixel_pro

if __name__=='__main__':
    parser = ArgumentParser()
    # general config
    parser.add_argument('--type_dataset', default='mvtec', type=str, choices=['mvtec', 'btad', 'visa'])
    parser.add_argument('--datadir', default='datasets/mvtec', type=str)
    parser.add_argument('--texture_source_dir', default='datasets/dtd/images', type=str)
    parser.add_argument('--imagesize', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int) # 4
    parser.add_argument('--seed', default=111, type=int) # 42
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--save_dir', default='saved_mvtec', type=str)
    # model config
    parser.add_argument('--nb_memory_sample', default=30, type=int)
    parser.add_argument('--l1_weight', default=0.6, type=float)
    parser.add_argument('--focal_weight', default=0.4, type=float)
    parser.add_argument('--focal_alpha', default=None)
    parser.add_argument('--focal_gamma', default=4, type=int)
    parser.add_argument('--lr', default=0.005, type=float) # 0.003
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--min_lr', default=0.0001, type=float)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--use_scheduler', default=True, type=bool)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
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
        best_img_auc, best_pixel_auc, best_pixel_pro = main(args)
        list_img_auc.append(best_img_auc)
        list_pixel_auc.append(best_pixel_auc)
        list_pixel_pro.append(best_pixel_pro)
    print('Average Image-level AUC: {:.4f}, Average Pixel-level AUC: {:.4f}, Average Pixel-level PRO: {:.4f}'.format(np.mean(list_img_auc), np.mean(list_pixel_auc), np.mean(list_pixel_pro)))