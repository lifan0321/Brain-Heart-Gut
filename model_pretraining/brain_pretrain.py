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
from data.GeneralDataset_brain import MyDataset
from models.transformers_cls_vit_brain import Transformer
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
    writer = SummaryWriter(log_dir=t_dir)
    
    # transform = transforms.Compose([transforms.ToTensor(),])
    transform = transforms.Compose([
        transforms.RandomRotation(10), #随机旋转图片
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    ])
    # 创建训练集和测试集的数据集对象和加载器

    train_dataset = MyDataset(root=args.data_dir, fold_path = args.fold_path, transform=transform, split='train')
    test_dataset = MyDataset(root=args.data_dir, fold_path = args.fold_path, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

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

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    AUC = ROCAUCMetric(average='macro')

    loss_epoch = CumulativeAverage()
    loss_epoch_final = CumulativeAverage()


    best_auc = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    # print loss and acc
    training_acc, training_loss, training_sen, training_spe, training_f1, training_auc = [], [], [], [], [], []
    testing_acc, testing_loss, testing_sen, testing_spe, testing_f1, testing_auc = [], [], [], [], [], []
    region_all = train_dataset.brain_91
    loss_epoch_all = [ CumulativeAverage() for _ in range(len(region_all))]
    training_acc_all, training_auc_all, training_loss_all = [[] for _ in range(len(region_all))], [[] for _ in range(len(region_all))], [[] for _ in range(len(region_all))]

    torch.autograd.set_detect_anomaly(True)
    num_epochs = 500
    for epoch in range(num_epochs): #200
        # print("epoch: ", epoch)
        model.train()

        label_pred = []
        label_real = []

        label_pred_all = [[] for _ in range(len(region_all))] 
        y_pred_all = [[] for _ in range(len(region_all))]
        y_all = [[] for _ in range(len(region_all))]        
        loss_epoch.reset()
        loss_epoch_final.reset()
        for loss_epoch_region in loss_epoch_all:
            loss_epoch_region.reset()

        AUC.reset()             
        for batch_idx, (inputs, lengths, labels) in enumerate(train_loader):
            inputs, lengths, labels = inputs.to('cuda'), lengths.to('cuda'), labels.to('cuda')
            optims.zero_grad()

            mciCount = torch.sum(labels == 0).item()
            ncCount = torch.sum(labels == 1).item()
            mciweight = ncCount / (mciCount + ncCount)
            ncweight = mciCount / (mciCount + ncCount)

            log_probs_region_all, fc1 = model(inputs, lengths)

            weights = [mciweight, ncweight]
            weights = [w if w != 0.0 else 1e-4 for w in weights]
            class_weights = torch.FloatTensor(weights).to('cuda')
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            outputs_align = log_probs_region_all[-1]
            loss = criterion(outputs_align, labels)
            loss_epoch_final.append(loss)

            for idx, log_probs_region in enumerate(log_probs_region_all[:-1]):
                loss_region = criterion(log_probs_region, labels)
                loss_epoch_all[idx].append(loss_region)
                label_pred_all[idx] += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(log_probs_region)]
                y_pred_all[idx] += [post_pred(i) for i in decollate_batch(log_probs_region)]
                y_all[idx] += [post_label(i) for i in decollate_batch(labels, detach=False)]
                loss += loss_region
            
            loss.backward()
            loss_epoch.append(loss)
            optims.step()

            label_real += [i for i in decollate_batch(labels)]
            label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs_align)]
            AUC(y_pred=[post_pred(i) for i in decollate_batch(outputs_align)],
                y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        loss_results = loss_epoch.aggregate()
        loss_final_results = loss_epoch_final.aggregate()
        cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger.print_message(f"Epoch :{int(epoch)} ")
        logger.print_message(f"Trainng    - Loss:{float(loss_results):.4f} ")
        logger.print_message(f"Align        - Loss_sub:{float(loss_final_results):.4f} "
                             f"ACC:{float(cm_train.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                             f"AUC:{AUC.aggregate():.4f}")

        writer.add_scalar("training_loss", float(loss_results), epoch)
        writer.add_scalar("training_acc", float(cm_train.Overall_ACC), epoch)
        writer.add_scalar("training_sen", float(list(cm_train.TPR.values())[1]), epoch)
        writer.add_scalar("training_spe", float(list(cm_train.TNR.values())[1]), epoch)
        writer.add_scalar("training_f1", float(list(cm_train.F1.values())[1]), epoch)
        writer.add_scalar("training_auc", float(AUC.aggregate()), epoch)
        training_loss.append(float(loss_results))
        training_acc.append(float(cm_train.Overall_ACC))
        training_sen.append(float(list(cm_train.TPR.values())[1]))
        training_spe.append(float(list(cm_train.TNR.values())[1]))
        training_f1.append(float(list(cm_train.F1.values())[1]))
        training_auc.append(float(AUC.aggregate()))

        for idx in range(len(region_all)):
            loss_results_region = loss_epoch_all[idx].aggregate()
            AUC.reset()
            AUC(y_pred=y_pred_all[idx], y=y_all[idx])
            cm_train_region = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred_all[idx])
            logger.print_message(f"{region_all[idx]}    - Loss:{loss_results_region} "
                                f"ACC:{float(cm_train_region.Overall_ACC):.4f} "
                                f"SEN:{float(list(cm_train_region.TPR.values())[1]):.4f} "
                                f"SPE:{float(list(cm_train_region.TNR.values())[1]):.4f} "
                                f"F1:{float(list(cm_train_region.F1.values())[1]):.4f} "
                                f"AUC:{AUC.aggregate():.4f}"
                                )
            training_acc_all[idx].append(float(cm_train_region.Overall_ACC))
            training_auc_all[idx].append(AUC.aggregate())
            training_loss_all[idx].append(loss_results_region)

        model.eval()
        loss_epoch.reset()
        AUC.reset()
        label_pred = []
        label_real = []

        with torch.no_grad():
            for batch_idx, (inputs, lengths, labels) in enumerate(test_loader):
                inputs, lengths, labels = inputs.to('cuda'), lengths.to('cuda'), labels.to('cuda')
                log_probs_region_all, fc1, = model(inputs, lengths)
                outputs = log_probs_region_all[-1]
                loss = F.cross_entropy(outputs, labels)            
                loss_epoch.append(loss)

                label_real += [i for i in decollate_batch(labels)]
                label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]

                AUC(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                    y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        loss_results = loss_epoch.aggregate()
        cm_test = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger.print_message(f"Validation    - Loss:{float(loss_results):.4f} "
                             f"ACC:{float(cm_test.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_test.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_test.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_test.F1.values())[1]):.4f} "
                             f"AUC:{AUC.aggregate():.4f}"
                             )

        writer.add_scalar("testing_loss", float(loss_results), epoch)
        writer.add_scalar("testing_acc", float(cm_test.Overall_ACC), epoch)
        writer.add_scalar("testing_sen", float(list(cm_test.TPR.values())[1]), epoch)
        writer.add_scalar("testing_spe", float(list(cm_test.TNR.values())[1]), epoch)
        writer.add_scalar("testing_f1", float(list(cm_test.F1.values())[1]), epoch)
        writer.add_scalar("testing_auc", float(AUC.aggregate()), epoch)
        testing_loss.append(float(loss_results))
        testing_acc.append(float(cm_test.Overall_ACC))

        testing_sen.append(float(list(cm_test.TPR.values())[1]))
        testing_spe.append(float(list(cm_test.TNR.values())[1]))
        testing_f1.append(float(list(cm_test.F1.values())[1]))
        testing_auc.append(float(AUC.aggregate()))
        
        auc = float(AUC.aggregate())
        if auc > best_auc:
            best_auc = auc
            logger.print_message("=====Find best model, saving checkpoint=======")
            outdict = model.state_dict()
            save_name = os.path.join(w_dir, "best_metric_model_.ckpt")
            torch.save(outdict, save_name)

    # print(training_acc)
    plot(training_acc, t_dir, folder_name + " training_acc")
    plot(training_loss, t_dir, folder_name + " training_loss")
    plot(training_sen, t_dir, folder_name + " training_sen")
    plot(training_spe, t_dir, folder_name + " training_spe")
    plot(training_f1, t_dir, folder_name + " training_f1")
    plot(training_auc, t_dir, folder_name + " training_auc")

    plot(testing_acc, t_dir, folder_name + " testing_acc")
    plot(testing_loss, t_dir, folder_name + " testing_loss")
    plot(testing_sen, t_dir, folder_name + " testing_sen")
    plot(testing_spe, t_dir, folder_name + " testing_spe")
    plot(testing_f1, t_dir, folder_name + " testing_f1")
    plot(testing_auc, t_dir, folder_name + " testing_auc")

    for idx in range(len(region_all)):
        plot(training_auc_all[idx], t_dir, folder_name + "_training_auc_" + region_all[idx])
        plot(training_acc_all[idx], t_dir, folder_name + "_training_acc_" + region_all[idx])
        plot(training_loss_all[idx], t_dir, folder_name + "_training_loss_" + region_all[idx])
if __name__ == "__main__":
    main()