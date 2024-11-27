import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from munch import Munch
from monai.utils import set_determinism
# from tqdm import tqdm
from data.GeneralDataset_GAT_align import BHGDataset
from models.GAT_align import GAT_align
from tensorboardX import SummaryWriter
from monai.data import decollate_batch
from pycm import ConfusionMatrix
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai import transforms
from monai.transforms import Compose, Activations, AsDiscrete
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json

parser = argparse.ArgumentParser(description="data loader for json")
parser.add_argument("--fold", default="=default", type=str, help="current fold")
parser.add_argument("--data_dir_brain", default="./", type=str, help="brain dataset directory")
parser.add_argument("--data_dir_all", default="./", type=str, help="all dataset directory")
parser.add_argument('--output', type=str, default='./output', help='Output folder for both tensorboard and the best model')
parser.add_argument("--pretrain_brain", default="xxx", type=str, help="pretrain brain")
parser.add_argument("--pretrain_all", default="xxx", type=str, help="pretrain all")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--hidden_layer_1", default=0, type=int, help="network hidden layer 1")
parser.add_argument("--hidden_layer_2", default=0, type=int, help="network hidden layer 2")
parser.add_argument("--save_file_name", default="1900-0101-0000", type=str, help="folder name to save subject")
parser.add_argument("--feature_size", default=0, type=int, help="dimension after pca")
parser.add_argument("--dropout", default="0.5", type=float, help="dropout for network")
parser.add_argument("--all_loss_weight", default="0.1", type=float, help="w1")
parser.add_argument("--sparse_brain_weight", default="0.2", type=float, help="w2")
parser.add_argument("--sparse_all_weight", default="0.1", type=float, help="w3")
parser.add_argument("--clip_feats_weight", default="0.2", type=float, help="w4")
parser.add_argument("--clip_corr_weight", default="0.1", type=float, help="w5")

def plot(data, dir, organ_name_type):
    try:
        x = list(range(len(data)))
        plt.plot(x, data, label=organ_name_type)
        plt.xlabel('epoch')
        plt.ylabel(organ_name_type)
        plt.title('Eval ' + organ_name_type)
        plt.xlim(0, len(data))
        plt.legend(loc='best')
        plt.savefig(dir + '/' + organ_name_type  + '.png')
        plt.close()
    except Exception as e:
        print(e)

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    print("Number of parameters of %s: %i" % (name, num_params))

class Logger():
    def __init__(self, log_dir, log_name='log.txt'):
        # create a logging file to store training losses
        self.log_name = os.path.join(log_dir, log_name)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg):
        # print(msg, flush=True)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

    def print_message_nocli(self, msg):
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

def save_metadata(metadata, output_path, output_postfix=''):
    """
    save the metadata of the dataset
    :return:
    """
    save = Compose([transforms.SaveImage(output_dir=output_path, output_postfix=output_postfix, print_log=True, resample=False, separate_folder=False), ])
    save(metadata)

def create_cos_target(indices, samplesize):
    matrix = torch.zeros(samplesize, samplesize).to('cuda')
    equal_elements = indices.unsqueeze(0) == indices.unsqueeze(1)
    matrix += equal_elements.float() * 1

    matrix.fill_diagonal_(1)
    return matrix

def main():
    seed = 20230329
    seed_torch(seed)

    args = parser.parse_args()
    root_dir = os.path.join(args.output)
    if os.path.exists(root_dir) == False:
        os.makedirs(root_dir, exist_ok=True)
        os.chmod(root_dir, 0o777)

    hidden_layer_1 = args.hidden_layer_1
    hidden_layer_2 = args.hidden_layer_2

    all_loss_weight = args.all_loss_weight
    sparse_brain_weight, sparse_all_weight = args.sparse_brain_weight, args.sparse_all_weight
    clip_feats_weight, clip_corr_weight = args.clip_feats_weight, args.clip_corr_weight
    
    folder_name = str(all_loss_weight) + "_" + str(sparse_brain_weight) + "_" + str(sparse_all_weight) + "_" + str(clip_feats_weight) + \
        "_" + str(clip_corr_weight)

    tensorboard_save_name = "tensorboard_" + args.save_file_name
    t_dir = os.path.join(root_dir, tensorboard_save_name)
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir, exist_ok=True)
        os.chmod(t_dir, 0o777)
    t_dir = os.path.join(t_dir, folder_name)
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir, exist_ok=True)
        os.chmod(t_dir, 0o777)
    t_dir = os.path.join(t_dir, "dropout_" + str(args.dropout))
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir, exist_ok=True)
        os.chmod(t_dir, 0o777)
    t_dir = os.path.join(t_dir, args.fold)
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir, exist_ok=True)
        os.chmod(t_dir, 0o777)


    weight_save_name = "weight_" + args.save_file_name
    w_dir = os.path.join(root_dir, weight_save_name)
    if os.path.exists(w_dir) == False:
        os.makedirs(w_dir, exist_ok=True)
        os.chmod(w_dir, 0o777)     
    w_dir = os.path.join(w_dir, folder_name)
    if os.path.exists(w_dir) == False:
        os.makedirs(w_dir, exist_ok=True)
        os.chmod(w_dir, 0o777)   
    w_dir = os.path.join(w_dir, "dropout_" + str(args.dropout))
    if os.path.exists(w_dir) == False:
        os.makedirs(w_dir, exist_ok=True)
        os.chmod(w_dir, 0o777) 
    w_dir = os.path.join(w_dir, args.fold)
    if os.path.exists(w_dir) == False:
        os.makedirs(w_dir, exist_ok=True)
        os.chmod(w_dir, 0o777) 


    train_dataset = BHGDataset(root_feats_brain=args.data_dir_brain, root_feats_all=args.data_dir_all, fold=args.fold, split='train')
    val_dataset = BHGDataset(root_feats_brain=args.data_dir_brain, root_feats_all=args.data_dir_all, fold=args.fold, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("============= hidden_layer_1 :" + str(hidden_layer_1) + " hidden_layer_2 : " + str(hidden_layer_2) + " batch_size: " + str(args.batch_size) + " fd: " + args.fold + " dropout " + str(args.dropout))

    pretrain_brain = args.pretrain_brain
    pretrain_all = args.pretrain_all 
    brain_region_size = 91
    all_region_size = 164
    gcnModel = GAT_align(brain_region_size, all_region_size,
                         args.feature_size, hidden_layer_1, hidden_layer_2, args.dropout,
                         args.pretrain_brain, args.pretrain_all)
    gcnModel.cuda()

    optims = torch.optim.AdamW(params=gcnModel.parameters(), lr=1e-03, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    AUC = ROCAUCMetric(average='macro')
    AUC_val = ROCAUCMetric(average='macro')
    AUC_test = ROCAUCMetric(average='macro')

    loss_epoch = CumulativeAverage()
    loss_sparse = CumulativeAverage()
    logger = Logger(t_dir, folder_name + '_log.txt')

    best_auc = 0.0

    training_acc, training_loss, training_sen, training_spe, training_f1, training_auc = [], [], [], [], [], []
    val_acc, val_sen, val_spe, val_f1, val_auc = [], [], [], [], []
    weight_clip_feats, weight_clip_corr, weight_sparse_brain, weight_sparse_all = None, None, None, None
    
    weight_sparse = None
    for epoch in range(300): #200
        gcnModel.train()  

        label_pred = []
        label_real = []   
        loss_epoch.reset()
        loss_sparse.reset()
        AUC.reset()
        for i, (inputs, corre, all_inputs, all_corre, labels) in enumerate(train_loader):
            optims.zero_grad()
            mciCount = torch.sum(labels == 0).item()
            ncCount = torch.sum(labels == 1).item()
            mciweight = ncCount / (mciCount + ncCount)
            ncweight = mciCount / (mciCount + ncCount)
            weights = [mciweight, ncweight]
            allCount = mciCount + ncCount
            cos_labels = create_cos_target(labels, allCount)

            brain_outputs, all_outputs, cosine_feats_mean_norm, cosine_corr_mean_norm, brain_attention, all_attention \
                = gcnModel(inputs, corre, all_inputs, all_corre)

            l1_norm = torch.norm(all_attention, p=1, dim = (1, 2))
            sparse_loss_all = torch.mean(l1_norm)
            if weight_sparse_all is None:
                weight_sparse_all = 10 ** ((-torch.floor(torch.log10(torch.tensor(sparse_loss_all))))-1)
            weight_sparse_all_loss = weight_sparse_all * sparse_loss_all

            l1_norm = torch.norm(brain_attention, p=1, dim = (1, 2))
            sparse_loss_brain = torch.mean(l1_norm)
            if weight_sparse_brain is None:
                weight_sparse_brain = 10 ** ((-torch.floor(torch.log10(torch.tensor(sparse_loss_brain))))-1)
            weight_sparse_brain_loss = weight_sparse_brain * sparse_loss_brain * 2

            loss_clip_feats = F.cross_entropy(cosine_feats_mean_norm, cos_labels)
            loss_clip_corr = F.cross_entropy(cosine_corr_mean_norm, cos_labels)

            if weight_clip_feats is None:
                weight_clip_feats = 10 ** ((-torch.floor(torch.log10(torch.tensor(loss_clip_feats)))) - 1)
            weight_loss_clip_feats = weight_clip_feats * loss_clip_feats

            if weight_clip_corr is None:
                weight_clip_corr = 10 ** ((-torch.floor(torch.log10(torch.tensor(loss_clip_corr)))) - 1)
            weight_loss_clip_corr = weight_clip_corr * loss_clip_corr

            class_weights = torch.FloatTensor(weights).to('cuda')
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            brain_loss = criterion(brain_outputs, labels)
            all_loss = criterion(all_outputs, labels)

            loss = 1 * brain_loss + all_loss_weight * all_loss + sparse_brain_weight * weight_sparse_brain_loss + \
                    sparse_all_weight * weight_sparse_all_loss + clip_feats_weight * weight_loss_clip_feats + clip_corr_weight * weight_loss_clip_corr

            loss.backward()
            loss_epoch.append(loss)
            optims.step()

            label_real += [i for i in decollate_batch(labels)]
            label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(brain_outputs)]
            AUC(y_pred=[post_pred(i) for i in decollate_batch(brain_outputs)],
                y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        loss_results = loss_epoch.aggregate()
        cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger.print_message(f"Epoch :{int(epoch)} ")
        logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} "
                             f"ACC:{float(cm_train.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                             f"AUC:{AUC.aggregate():.4f}")

        training_loss.append(float(loss_results))
        training_acc.append(float(cm_train.Overall_ACC))
        training_sen.append(float(list(cm_train.TPR.values())[1]))
        training_spe.append(float(list(cm_train.TNR.values())[1]))
        training_f1.append(float(list(cm_train.F1.values())[1]))
        training_auc.append(float(AUC.aggregate()))

        gcnModel.eval()  
        AUC_val.reset()
        label_pred = []
        label_real = []                  
        with torch.no_grad():
            for batch_idx, (inputs, corre, all_inputs, all_corre, labels) in enumerate(val_loader):
                outputs, _, _, _, _, _  = gcnModel(inputs, corre, all_inputs, all_corre)

                label_real += [i for i in decollate_batch(labels)]
                label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]
                AUC_val(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                    y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger.print_message(f"Validation "
                             f"ACC:{float(cm_val.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_val.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_val.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_val.F1.values())[1]):.4f} "
                             f"AUC:{AUC_val.aggregate():.4f}"
                             )     

        val_acc.append(float(cm_val.Overall_ACC))
        val_sen.append(float(list(cm_val.TPR.values())[1]))
        val_spe.append(float(list(cm_val.TNR.values())[1]))
        val_f1.append(float(list(cm_val.F1.values())[1]))
        val_auc.append(float(AUC_val.aggregate()))

        auc = float(AUC_val.aggregate())
        if auc >= best_auc:
            best_auc = auc
            save_name = os.path.join(w_dir, "best_auc.ckpt")
            print('Saving best auc into %s...' % save_name)
            outdict = gcnModel.state_dict()
            torch.save(outdict, save_name)


if __name__ == "__main__":
    main()