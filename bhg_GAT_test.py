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
from sklearn.metrics import roc_curve, auc

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
parser.add_argument("--weightInput", default="xxx/xxxx/", type=str, help="pretrain_file for weights")

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


class Logger():
    def __init__(self, log_dir, log_name='log.txt'):
        # create a logging file to store training losses
        self.log_name = os.path.join(log_dir, log_name)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg):
        print(msg, flush=True)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

    def print_message_nocli(self, msg):
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

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

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    AUC_epoch = ROCAUCMetric(average='macro')
    logger = Logger(t_dir, folder_name + '_log.txt')
    
    test_dataset = MyDataset(root=args.data_dir, fold=str(args.fold), split='train')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(str(hidden_layer_1) + " " + str(hidden_layer_2) + " fd: " + str(args.fold), flush=True)
    gcnModel = GAT_novel(args.region_size, args.feature_size, hidden_layer_1, hidden_layer_2, args.dropout)
    gcnModel.cuda()

    weights = args.weightInput
    module_dict = torch.load(weights)   
    gcnModel.load_state_dict(module_dict)         
    
    gcnModel.eval()  
    label_pred_epoch = []
    label_real_epoch = []
    AUC_epoch.reset()
    with torch.no_grad():
        for batch_idx, (inputs, corre, labels) in enumerate(test_loader):
            outputs, attention = gcnModel(inputs, corre)

            label_real_epoch += [i for i in decollate_batch(labels)]
            label_pred_epoch += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]

            AUC_epoch(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                y=[post_label(i) for i in decollate_batch(labels, detach=False)])

    cm_test = ConfusionMatrix(actual_vector=label_real_epoch, predict_vector=label_pred_epoch)
    logger.print_message(f"each epoch "
                        f"ACC:{float(cm_test.Overall_ACC):.4f} "
                        f"SEN:{float(list(cm_test.TPR.values())[1]):.4f} "
                        f"SPE:{float(list(cm_test.TNR.values())[1]):.4f} "
                        f"F1:{float(list(cm_test.F1.values())[1]):.4f} "
                        f"AUC_epoch:{AUC_epoch.aggregate():.4f}"
                        )



if __name__ == "__main__":
    main()