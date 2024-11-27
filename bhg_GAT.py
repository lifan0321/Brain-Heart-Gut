import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from munch import Munch
from monai.utils import set_determinism
from data.GeneralDataset_GAT import MyDataset
from models.GAT_novel import GAT_novel

from tensorboardX import SummaryWriter
from monai.data import decollate_batch
from pycm import ConfusionMatrix
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai import transforms
from monai.transforms import Compose, Activations, AsDiscrete
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

parser = argparse.ArgumentParser(description="data loader for json")
parser.add_argument("--fold", default="=default", type=str, help="current fold")
parser.add_argument("--data_dir", default="./", type=str, help="dataset directory")
parser.add_argument('--output', type=str, default='./output', help='Output folder for both tensorboard and the best model')
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--hidden_layer_1", default=0, type=int, help="network hidden layer 1")
parser.add_argument("--hidden_layer_2", default=0, type=int, help="network hidden layer 2")
parser.add_argument("--save_file_name", default="1900-0101-0000", type=str, help="folder name to save subject")
parser.add_argument("--region_size", default=0, type=int, help="the size of region: brain 99, gut 50, heart 21")
parser.add_argument("--feature_size", default=0, type=int, help="dimension after pca")
parser.add_argument("--dropout", default="0.5", type=float, help="dropout for network")

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

def custom_loss(attention, zero_ratio):
    num_elements = attention.size(1)
    num_zero_elements = int(zero_ratio * num_elements)
    row_min_elements, _ = torch.topk(torch.abs(attention), num_zero_elements, dim=2, largest=False)
    row_loss = torch.sum(row_min_elements ** 2)
    
    col_min_elements, _ = torch.topk(torch.abs(attention), num_zero_elements, dim=1, largest=False)
    col_loss = torch.sum(col_min_elements ** 2)
    
    return row_loss + col_loss

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
    folder_name = str(hidden_layer_1) + "_" + str(hidden_layer_2) + "_" + str(args.batch_size)

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

    # 创建训练集和测试集的数据集对象和加载器
   
    train_dataset = MyDataset(root=args.data_dir, fold=args.fold, split='train')
    val_dataset = MyDataset(root=args.data_dir, fold=args.fold, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(str(hidden_layer_1) + " " + str(hidden_layer_2) + " fd: " + args.fold, flush=True)
    gat = GAT_novel(args.region_size, args.feature_size, hidden_layer_1, hidden_layer_2, args.dropout)
    gat.cuda()

    optims = torch.optim.AdamW(params=gat.parameters(), lr=1e-03, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
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
    
    for epoch in range(500): #300
        gat.train()  

        label_pred = []
        label_real = []   
        loss_epoch.reset()
        loss_sparse.reset()
        AUC.reset()
        for i, (inputs, corre, labels) in enumerate(train_loader):
            optims.zero_grad()

            mciCount = torch.sum(labels == 0).item()
            ncCount = torch.sum(labels == 1).item()
            mciweight = ncCount / (mciCount + ncCount)
            ncweight = mciCount / (mciCount + ncCount)

            outputs, attention = gat(inputs, corre)

            weights = [mciweight, ncweight]
            class_weights = torch.FloatTensor(weights).to('cuda')
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(outputs, labels)

            weight_sparse_loss = 0.01 * custom_loss(attention, 0.6)
            loss += weight_sparse_loss
            loss.backward()
            loss_epoch.append(loss)
            loss_sparse.append(weight_sparse_loss)
            optims.step()
            label_real += [i for i in decollate_batch(labels)]
            label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]
            AUC(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        loss_results = loss_epoch.aggregate()
        loss_results_sparse = loss_sparse.aggregate()
        cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger.print_message(f"Epoch :{int(epoch)} ")
        logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} "
                             f"ACC:{float(cm_train.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                             f"AUC:{AUC.aggregate():.4f}")
        logger.print_message(f"loss_results_sparse :{float(loss_results_sparse):.4f} ")

        training_loss.append(float(loss_results))
        training_acc.append(float(cm_train.Overall_ACC))
        training_sen.append(float(list(cm_train.TPR.values())[1]))
        training_spe.append(float(list(cm_train.TNR.values())[1]))
        training_f1.append(float(list(cm_train.F1.values())[1]))
        training_auc.append(float(AUC.aggregate()))

        gat.eval()  
        AUC_val.reset()
        label_pred = []
        label_real = []                  
        with torch.no_grad():
            for batch_idx, (inputs, corre, labels) in enumerate(val_loader):
                outputs, attention = gat(inputs, corre)

                label_real += [i for i in decollate_batch(labels)]
                label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]
                AUC_val(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                    y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger.print_message(f"Validation    - "
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
            outdict = gat.state_dict()
            torch.save(outdict, save_name)

    plot(training_acc, t_dir, folder_name + " training_acc")
    plot(training_loss, t_dir, folder_name + " training_loss")
    plot(training_sen, t_dir, folder_name + " training_sen")
    plot(training_spe, t_dir, folder_name + " training_spe")
    plot(training_f1, t_dir, folder_name + " training_f1")
    plot(training_auc, t_dir, folder_name + " training_auc")
    plot(val_acc, t_dir, folder_name + " val_acc")
    plot(val_sen, t_dir, folder_name + " val_sen")
    plot(val_spe, t_dir, folder_name + " val_spe")
    plot(val_f1, t_dir, folder_name + " val_f1")
    plot(val_auc, t_dir, folder_name + " val_auc")  

if __name__ == "__main__":
    main()