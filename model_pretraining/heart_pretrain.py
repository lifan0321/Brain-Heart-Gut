import torch
import torch.cuda as cuda
import random
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from munch import Munch
from monai.utils import set_determinism
# from tqdm import tqdm
from data.GeneralDataset_recon import MyDataset
from models.transformers_cls_vit_heart import Transformer
from tensorboardX import SummaryWriter
from monai.data import decollate_batch
from pycm import ConfusionMatrix
from monai.metrics import CumulativeAverage, ROCAUCMetric
# from monai import transforms
from monai.transforms import Compose, Activations, AsDiscrete
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

parser = argparse.ArgumentParser(description="data loader for json")
parser.add_argument('--output', type=str, default='./output', help='Output folder for both tensorboard and the best model')
parser.add_argument("--data_dir", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold_path", default="dataset_0.json", type=str, help="fold for dataset")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--num_head", default=1, type=int, help="num_head of transformer")
parser.add_argument("--num_layers", default=1, type=int, help="num_layers for transformer")
parser.add_argument("--dim_feedforward", default=1, type=int, help="dim_feedforward for transformer")
parser.add_argument("--hidden_dim_fc", default=1, type=int, help="hidden_dim_fc for transformer")
parser.add_argument("--save_file_name", default="1900-0101-0000", type=str, help="folder name to save subject")
parser.add_argument("--dropout", default="0.5", type=float, help="dropout for network")

def plot(data, dir, organ_name_type):
    try:
        x = list(range(len(data)))
        plt.plot(x, data, label=organ_name_type)
        plt.xlabel('epoch 1')
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
    torch.backends.cudnn.deterministic = True

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
        print(msg, flush=True)
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

def collate_fn(data_labels):
    region_num = len(data_labels[0][0])
    subject_num = len(data_labels)
    # print(region_num, subject_num)
    batch_inputs = []
    batch_lengths = []
    targets = []
    for region_index in range(region_num):
        targets_batch = []
        for subject_index in range(subject_num):
            batch_inputs.append(data_labels[subject_index][0][region_index])
            targets_batch.append(data_labels[subject_index][1][region_index])
            batch_lengths.append(len(data_labels[subject_index][0][region_index]))
        tensor = torch.stack(targets_batch)
        # print(tensor.size())
        targets.append(tensor)
        del tensor

    batch_inputs = pad_sequence(batch_inputs, batch_first=True)
    _, patch_num, dim = batch_inputs.size()
    batch_inputs = batch_inputs.view(-1, len(data_labels), patch_num, dim)

    batch_lengths = torch.tensor(batch_lengths)
    batch_lengths = batch_lengths.view(-1, len(data_labels))

    del data_labels
    return batch_inputs, batch_lengths, targets

def memorycheck():
    props = cuda.get_device_properties(0)
    total_memory = props.total_memory // 1024**2  # 转换为MB
    free_memory = cuda.memory_allocated(0) // 1024**2  # 转换为MB
    print("GPU 0 总内存: {}MB".format(total_memory), flush=True)
    print("GPU 0 可用内存: {}MB".format(free_memory), flush=True)

def main():
    seed = 20230329
    seed_torch(seed)

    args = parser.parse_args()
    root_dir = os.path.join(args.output)
    if os.path.exists(root_dir) == False:
        os.makedirs(root_dir)

    folder_name = str(args.num_head) + "_" + str(args.num_layers) + "_" + str(args.batch_size) + "_" + str(args.dropout) + "_" + str(args.dim_feedforward) + "_" + str(args.hidden_dim_fc)
    tensorboard_save_name = "tensorboard_" + args.save_file_name
    t_dir = os.path.join(root_dir, tensorboard_save_name, folder_name, "dropout_" + str(args.dropout))
    if os.path.exists(t_dir) == False:
        os.makedirs(t_dir)
    weight_save_name = "weight_" + args.save_file_name
    w_dir = os.path.join(root_dir, weight_save_name, folder_name, "dropout_" + str(args.dropout))
    if os.path.exists(w_dir) == False:
        os.makedirs(w_dir)    
    
    # transform = transforms.Compose([transforms.ToTensor(),])
    transform = transforms.Compose([
        transforms.RandomRotation(10), #随机旋转图片
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    ])
    # 创建训练集和测试集的数据集对象和加载器

    train_dataset = MyDataset(root=args.data_dir, fold_path = args.fold_path, transform=transform, split='train')
    test_dataset = MyDataset(root=args.data_dir, fold_path = args.fold_path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    print("============= num_layers :" + str(args.num_layers) + " num_head : " + str(args.num_head) + " batch_size: " + str(args.batch_size) + " dropout " + str(args.dropout))
    embedding_dim = 512
    dim_feedforward = args.dim_feedforward
    hidden_dim_fc = args.hidden_dim_fc
    dim_feedforward_concate = 32
    hidden_dim_fc_concate = 16
    num_class = 2
    num_head = args.num_head
    num_layers = args.num_layers
    dropout=args.dropout
    model = Transformer(embedding_dim, dim_feedforward, hidden_dim_fc,
                        dim_feedforward_concate, hidden_dim_fc_concate,
                        num_class, num_head, num_layers,
                        dropout)

    model.cuda()

    logger = Logger(t_dir, folder_name + '_log.txt')
    optims = torch.optim.AdamW(params=model.parameters(), lr=1e-04, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)

    loss_epoch = CumulativeAverage()

    best_auc = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    training_acc, training_loss, training_sen, training_spe, training_f1, training_auc = [], [], [], [], [], []
    testing_acc, testing_loss, testing_sen, testing_spe, testing_f1, testing_auc = [], [], [], [], [], []

    torch.autograd.set_detect_anomaly(True)
    num_epochs = 2000
    for epoch in range(num_epochs): #200
        model.train()
        loss_epoch.reset()

        for batch_idx, (inputs, lengths, labels) in enumerate(train_loader):
            inputs, lengths = inputs.to('cuda'), lengths.to('cuda')
            labels = [matrix.to('cuda') for matrix in labels]
            optims.zero_grad()

            log_probs_region_all = model(inputs, lengths)

            criterion = nn.L1Loss()
            loss = 0
            for idx, log_probs_region in enumerate(log_probs_region_all):
                loss_region = criterion(log_probs_region, labels[idx])
                loss += loss_region
            
            loss.backward()
            
            loss_epoch.append(loss)
            optims.step()

        loss_results = loss_epoch.aggregate()
        logger.print_message(f"Epoch :{int(epoch)} ")
        logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} ")

        training_loss.append(float(loss_results))

        model.eval()
        loss_epoch.reset()

        with torch.no_grad():
            for batch_idx, (inputs, lengths, labels) in enumerate(test_loader):
                inputs, lengths = inputs.to('cuda'), lengths.to('cuda')
                labels = [matrix.to('cuda') for matrix in labels]

                log_probs_region_all= model(inputs, lengths)

                criterion = nn.L1Loss()
                loss = 0
                for idx, log_probs_region in enumerate(log_probs_region_all):
                    loss_region = criterion(log_probs_region, labels[idx])
                    loss += loss_region
                loss_epoch.append(loss)


        loss_results = loss_epoch.aggregate()
        logger.print_message(f"Validation    - Loss:{float(loss_results):.4f} "
                             )

        testing_loss.append(float(loss_results))

        outdict = model.state_dict()
        save_name = os.path.join(w_dir, "metric_model_" + str(epoch + 1) + ".ckpt")
        torch.save(outdict, save_name)


    plot(training_loss, t_dir, folder_name + " training_loss")

    plot(testing_loss, t_dir, folder_name + " testing_loss")

if __name__ == "__main__":
    main()