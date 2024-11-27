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
from data.GeneralDataset_brain_heart_gut import MyDataset
from models.transformers_cls_vit_bhg import Transformer
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
parser.add_argument("--root_feats_brain", default="xxx", type=str, help="brain path")
parser.add_argument("--root_feats_heart", default="xxx", type=str, help="heart path")
parser.add_argument("--root_feats_gut", default="xxx", type=str, help="gut path")
parser.add_argument("--fold", default="=default", type=str, help="current fold")
parser.add_argument('--output', type=str, default='./output', help='Output folder for both tensorboard and the best model')
parser.add_argument("--fold_path", default="dataset_0.json", type=str, help="fold for dataset")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--dim_feedforward", default=1, type=int, help="dim_feedforward for transformer")
parser.add_argument("--hidden_dim_fc", default=1, type=int, help="hidden_dim_fc for transformer")
parser.add_argument("--save_file_name", default="1900-0101-0000", type=str, help="folder name to save subject")
parser.add_argument("--dropout", default="0.5", type=float, help="dropout for network")
parser.add_argument("--weightInput", default="xxx/xxxx/", type=str, help="pretrain_file for weights")

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
    batch_inputs = []
    batch_lengths = []

    for region_index in range(region_num):
        for subject_index in range(subject_num):
            batch_inputs.append(data_labels[subject_index][0][region_index])
            batch_lengths.append(len(data_labels[subject_index][0][region_index]))
    
    batch_inputs = pad_sequence(batch_inputs, batch_first=True)
    _, patch_num, dim = batch_inputs.size()
    batch_inputs = batch_inputs.view(-1, len(data_labels), patch_num, dim)
        
    targets = torch.tensor([ex[1] for ex in data_labels], dtype=torch.long)
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

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    AUC = ROCAUCMetric(average='macro')
    logger = Logger(t_dir, folder_name + '_log.txt')

    filepath = args.task + "/output/weight_" + args.seed + "/4_4_8_0.5_216_128/dropout_0.5"
    print("============= num_layers :" + str(args.num_layers) + " num_head : " + str(args.num_head) + " batch_size: " + str(args.batch_size) + " dropout " + str(args.dropout))

    test_dataset = MyDataset(root_feats_brain = args.root_feats_brain, root_feats_heart = args.root_feats_heart, root_feats_gut = args.root_feats_gut,
                            fold_path = args.fold_path, fold=args.fold, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

    embedding_dim = 512
    dim_feedforward = args.dim_feedforward
    hidden_dim_fc = args.hidden_dim_fc
    dim_feedforward_concate = 32
    hidden_dim_fc_concate = 16
    num_class = 2

    model = Transformer(embedding_dim, dim_feedforward, hidden_dim_fc,
                        dim_feedforward_concate, hidden_dim_fc_concate,
                        num_class)

    model.cuda()

    label_pred = []
    label_real = []
    AUC.reset()

    weights = args.weightInput
    module_dict = torch.load(weights)   
    model.load_state_dict(module_dict)

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, lengths, labels) in enumerate(test_loader):
            inputs, lengths, labels = inputs.to('cuda'), lengths.to('cuda'), labels.to('cuda')
            log_probs_region_all, fc1, _, _= model(inputs, lengths)
            outputs = log_probs_region_all[-1]

            label_real += [i for i in decollate_batch(labels)]
            label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]

            AUC(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                y=[post_label(i) for i in decollate_batch(labels, detach=False)])

    cm_test = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
    logger.print_message(f"Validation "
                        f"ACC:{float(cm_test.Overall_ACC):.4f} "
                        f"SEN:{float(list(cm_test.TPR.values())[1]):.4f} "
                        f"SPE:{float(list(cm_test.TNR.values())[1]):.4f} "
                        f"F1:{float(list(cm_test.F1.values())[1]):.4f} "
                        f"AUC:{AUC.aggregate():.4f}"
                        )


if __name__ == "__main__":
    main()