import os
import time
import datetime
import h5py
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support
from lib.dataset.data_set import *
from lib.configs.parse_arg import opt, args
from lib.network.mynn import Upsample
from lib.network.deepv3 import *
from lib.utils import *
from lib.utils.metric import *
from lib.utils.img_utils import Compose, Normalize, ToTensor, Resize, Distortion, ResizeImg, Fog, ColorJitter, GaussianBlur
import lib.loss as loss
from lib.network.mynn import PatchNorm2d
from lib.configs.parse_arg import opt, args
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OOD_Model(object):
    def __init__(self, method):
        super(OOD_Model, self).__init__()
        self.since = time.time()
        self.log_init()
        self.phases = []

        if not args.trans_type:
            trans_type = opt.data.trans_type
        else:
            trans_type = args.trans_type

        self.data_loaders ={
            'test': DataLoader(self.build_dataset(trans_type=trans_type), batch_size=opt.train.test_batch, drop_last=False,
                                num_workers=0, shuffle=False, pin_memory=True)}
        self.method = method    #num_workers=opt.data.num_workers
        self.best = {}

    """
    Prepare dataloader, params, optimizer, loss etc.
    """
    def build_dataset(self, dataset = None, trans_type = 'test'):
        # data transformation
        m_transforms = {
            # Testing Transformation
            'test': Compose([
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            # Transformation for adding Domain-Shift
            'distortion': Compose([
                Distortion(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            'color_jitter':Compose([
                ColorJitter(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            'gaussian_blur': Compose([
                GaussianBlur(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            'fog': Compose([
                Fog(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
        }

        if not dataset:
            dataset = args.dataset

        # Inlier dataset
        if dataset == "Cityscapes_train":
            ds = Cityscapes(split="train", transform=m_transforms[trans_type])
        elif dataset == "Cityscapes_val":
            ds = Cityscapes(split="val", transform=m_transforms[trans_type])

        # Anomaly dataset
        elif dataset == "RoadAnomaly":
            ds = RoadAnomaly(transform=m_transforms['test'])
        elif dataset == "FS_LostAndFound":
            ds = Fishyscapes(split="LostAndFound", transform=m_transforms['test'])
        elif dataset == "FS_Static":
            ds = Fishyscapes(split="Static", transform=m_transforms['test'])
        elif dataset == "RoadAnomaly21":
            ds = RoadAnomaly21(transform=m_transforms['test'])
        elif dataset == "RoadObstacle21":
            ds = RoadObstacle21(transform=m_transforms['test'])

        # Constructed Dataset with Domain-Shift
        elif dataset == "FS_Static_C":
            ds = Fishyscapes(split="Static", transform=m_transforms['distortion'], domain_shift=False)
        elif dataset == "FS_Static_C_sep":
            ds = Fishyscapes(split="Static", transform=m_transforms[trans_type], domain_shift=False)
        else:
            self.logger.warning("No dataset!")

        print(ds, len(ds))
        return ds

    def configure_trainable_params(self, trainable_params_name):
        self.method.model.requires_grad_(False)
        params = []
        names = []
        for nm, m in self.method.model.named_modules():
            if (trainable_params_name == 'bn' and isinstance(m, nn.BatchNorm2d))\
            or (trainable_params_name != 'bn' and trainable_params_name in nm):
                for np, p in m.named_parameters():
                    if f"{nm}.{np}" not in names:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def set_sbn_momentum(self, momentum = None):
        for name, module in self.method.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = momentum

    def configure_bn(self, momentum = 0.0):
        for nm, m in self.method.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if opt.train.instance_BN:
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                else:
                    m.adapt = True
                    m.momentum = momentum
        return

    def build_optimizer(self, params, lr):
        if opt.train.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay = opt.train.weight_decay)
        elif opt.train.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=opt.train.momentum)

        return optimizer

    def build_loss(self):
        Criterion = getattr(loss, opt.loss.name)
        criterion = Criterion(opt.loss.params)
        return criterion

    def tta_preparation(self):
        if opt.model.trainable_params_name is not None:
            params, names = self.configure_trainable_params(opt.model.trainable_params_name)
            self.optimizer = self.build_optimizer(params, opt.train.lr)
            self.criterion = self.build_loss()

        if opt.train.episodic: # Store Initial Model and Optimizer Stat.
            self.initial_model_state = deepcopy(self.method.model.state_dict())
            self.initial_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.configure_bn()

        return


    """
    Main functions for test time adaptation.
    """

    def atta(self, img, i, ret_logit =False):
        # We use Episodic Training Manner by default.
        if opt.train.episodic:
            self.method.model.load_state_dict(self.initial_model_state)
            self.optimizer.load_state_dict(self.initial_optimizer_state)
            self.configure_bn()

        # 1. Selective Bacth Normalization
        with torch.no_grad():
            ds_prob = self.get_domainshift_prob(img, i)
            #self.set_sbn_momentum(momentum=ds_prob)
            self.set_sbn_momentum(momentum=ds_prob)

        # 2. Anomaly-aware Self-Training
        anomaly_score, logit = self.method.anomaly_score(img, ret_logit=True)
        loss = self.criterion(anomaly_score, logit)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Forward to get the final output.
        with torch.no_grad():
            # Define the folder where you want to save the logit tensor
            save_folder = './saved_data/logit_tensors'
            save_path = os.path.join(save_folder, 'logit_tensor.pt')
            os.makedirs(save_folder, exist_ok=True)

            # Since only the final block params are updated, we only need to recalculate for this block.
            feature = self.method.model.module.final(self.method.model.module.dec0)
            logit = Upsample(feature, img.size()[2:])
            anomaly_score = self.method.getscore_from_logit(logit)

            torch.save(logit, save_path)

        if ret_logit:
            return anomaly_score, logit
        return anomaly_score

    def tent(self, img, ret_logit =False):
        anomaly_score, logit = self.method.anomaly_score(img, ret_logit=True)
        loss = self.criterion(anomaly_score, logit)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            anomaly_score, logit = self.method.anomaly_score(img, ret_logit=True)
        if ret_logit:
            return anomaly_score, logit
        return anomaly_score

    """
    Inference function for detect unknown classes.
    """

    def inference(self):
        self.method.model.train(False)
        if opt.train.tta is not None:
            self.tta_preparation()
        anomaly_score_list = []
        ood_gts_list = []

        for (i,data) in tqdm(enumerate(self.data_loaders['test'])):
            img, target = data[0].to(device), data[1].numpy()
            if opt.train.tta == 'atta':
                anomaly_score = self.atta(img,i)
            elif opt.train.tta == 'tent':
                anomaly_score = self.tent(img)
            else:
                anomaly_score = self.method.anomaly_score(img)
            anomaly_npy = anomaly_score.detach().cpu().numpy()
            #self.calculate_metrcis(target, anomaly_npy, i) # Uncomment this for debuggging.
            ood_gts_list.append(target)
            anomaly_score_list.append(anomaly_npy)

        # Convert lists to numpy arrays
        ood_gts_array = np.array(ood_gts_list)
        anomaly_score_array = np.array(anomaly_score_list)

        # Save the arrays as .npy files
        save_folder = f'./saved_data/anomaly_results/{args.dataset}/{args.patch_div}_{args.patch_div}/'
        os.makedirs(save_folder, exist_ok=True)
        
        np.save(os.path.join(save_folder, 'ood_gts_list.npy'), ood_gts_array)
        np.save(os.path.join(save_folder, 'anomaly_score_list.npy'), anomaly_score_array)

        roc_auc, prc_auc, fpr95 = eval_ood_measure(np.array(anomaly_score_list), np.array(ood_gts_list))
        logging.warning(f'AUROC score for {args.dataset}: {roc_auc:.2%}')
        logging.warning(f'AUPRC score for {args.dataset}: {prc_auc:.2%}')
        logging.warning(f'FPR@TPR95 for {args.dataset}: {fpr95:.2%}')

    """
    Inference function for known class evaluation measures.
    """

    def inference_known(self):
        self.method.model.train(False)
        all_results = []
        with torch.no_grad():
            for (i,data) in tqdm(enumerate(self.data_loaders['test'])):
                img, ood_gts, target = data[0].to(device), data[1].long(), data[2].long()
                outputs = self.method.anomaly_score(img, ret_logit = True)[1]
                pred = outputs.argmax(1).detach().cpu().numpy()
                ood_gts = ood_gts.numpy()
                target = target.numpy()

                label_inlier = target[(ood_gts == 0) & (target != 255)]
                pred_inlier = pred[(ood_gts == 0) & (target != 255)]

                hist_tmp, labeled_tmp, correct_tmp = hist_info(19, pred_inlier, label_inlier)
                results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
                all_results.append(results_dict)
        m_iou, m_acc = compute_metric(all_results)

        with open(f'./saved_data/predictions/{args.dataset}/all_results.json', 'w') as file:
            json.dump(all_results, file)

        logging.warning("current mIoU is {}, mAcc is {}".format(m_iou, m_acc))
        return


    """
    Functions related to BN-based domain shift detection.
    """
    ################### BN
    def get_domainshift_prob(self, x, i, threshold = 50.0, beta = 0.1, epsilon = 1e-8):
        if args.save_img:
            if args.trans_type == 'fog':
                img_folder = f'./saved_data/input_images/{args.dataset}/fog'
            else:
                img_folder = f'./saved_data/input_images/{args.dataset}/no_fog'
            os.makedirs(img_folder, exist_ok=True)
            torch.save(x,f'{img_folder}/image{i}.pth')

        if args.anomalies:
            if args.custom_bn:
                if args.trans_type == 'fog':
                    kl_folder = f'./saved_data/kl/{args.dataset}/patch/anomalies/fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/patch/anomalies/fog'
                else:
                    kl_folder = f'./saved_data/kl/{args.dataset}/patch/anomalies/no_fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/patch/anomalies/no_fog'
            else:
                if args.trans_type == 'fog':
                    kl_folder = f'./saved_data/kl/{args.dataset}/global/anomalies/fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/global/anomalies/fog'
                else:
                    kl_folder = f'./saved_data/kl/{args.dataset}/global/anomalies/no_fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/global/anomalies/no_fog'
        else:
            if args.custom_bn:
                if args.trans_type == 'fog':
                    kl_folder = f'./saved_data/kl/{args.dataset}/patch/no_anomalies/fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/patch/no_anomalies/fog'
                else:
                    kl_folder = f'./saved_data/kl/{args.dataset}/patch/no_anomalies/no_fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/patch/no_anomalies/no_fog'
            else:
                if args.trans_type == 'fog':
                    kl_folder = f'./saved_data/kl/{args.dataset}/global/no_anomalies/fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/global/no_anomalies/fog'
                else:
                    kl_folder = f'./saved_data/kl/{args.dataset}/global/no_anomalies/no_fog'
                    proba_folder = f'./saved_data/probabilities/{args.dataset}/global/no_anomalies/no_fog'

        # Create the directories if they do not exist
        os.makedirs(kl_folder, exist_ok=True)
        os.makedirs(proba_folder, exist_ok=True)

        # Perform forward propagation
        self.method.anomaly_score(x)

        # Calculate the aggregated discrepancy
        if args.custom_bn:
            discrepancy = None
        else:
            discrepancy = torch.zeros((1,1,1), device=device)
        for name, layer in self.method.model.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                print('batchnorm layer: ', name)
                mu_x, var_x = layer.mean, layer.var

                if mu_x.dim() == 3:
                    mu_x = mu_x[None,:,:,:]
                    var_x = var_x[None,:,:,:]

                mu, var = layer.running_mean, layer.running_var

                # If it's the first iteration where you encounter a matching layer, initialize discrepancy
                if discrepancy is None:
                    B,C,h,w = mu_x.shape
                    discrepancy = torch.zeros((B,h,w), device=device)
                    #print('size discrep:', discrepancy.shape)

                if mu_x.shape != mu.shape:
                    #print('discrep diff sizes')
                    expanded_mu = mu.view(1, -1, 1, 1)
                    expanded_var = var.view(1, -1, 1, 1)
                    discrepancy = discrepancy + 0.5 * (torch.log((expanded_var + epsilon) / (var_x + epsilon)) + (var_x + (mu_x - expanded_mu) ** 2) / (expanded_var + epsilon) - 1).sum(dim=1)
                    kl_elem = 0.5 * (torch.log((expanded_var + epsilon) / (var_x + epsilon)) + (var_x + (mu_x - expanded_mu) ** 2) / (expanded_var + epsilon) - 1).sum(dim=1)
                else:
                    discrepancy = discrepancy + 0.5 * (torch.log((var + epsilon) / (var_x + epsilon)) + (
                                var_x + (mu_x - mu) ** 2) / (var + epsilon) - 1).sum()
                    kl_elem = 0.5 * (torch.log((var + epsilon) / (var_x + epsilon)) + (
                                var_x + (mu_x - mu) ** 2) / (var + epsilon) - 1).sum()
                    #print('discrep:', discrepancy)
                print('added kl:', kl_elem)
                if args.save_stats:
                    torch.save(kl_elem,f'{kl_folder}/kl_{name}_img{i}.pth')
                print(f'cumul discrepancy at layer {name}: {discrepancy}')
        # Training Data Stat. (Use function 'save_bn_stats' to obtain for different models).
        if opt.model.backbone == 'WideResNet38':
            train_stat_mean = 825.3230302274227
            train_stat_std = 131.76657988963967
        elif opt.model.backbone == 'ResNet101':
            train_stat_mean = 2428.9796256740888
            train_stat_std = 462.1095033939578

        # Normalize KL Divergence to a probability.
        discrepancy = discrepancy.squeeze()
        #print('discrep shape before norm:', discrepancy.shape, 'discrep:', discrepancy)
        normalized_kl_divergence_values = (discrepancy - train_stat_mean) / train_stat_std
        if args.save_stats:
            save_dir = f'{proba_folder}_before_sigmoid/8_16'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(normalized_kl_divergence_values,f'{save_dir}/momentum_img{i}.pth')
        print('before sig:', normalized_kl_divergence_values)
        momentum = torch.sigmoid(beta * (normalized_kl_divergence_values - threshold))
        #momentum = torch.sigmoid(normalized_kl_divergence_values)
        #momentum = torch.tensor(0)
        print('calculated momentum:', momentum)
        #if args.save_stats:
        #    torch.save(momentum,f'{proba_folder}/proba_img{i}.pth')
        return momentum



    def save_bn_stats(self):
        self.method.model.train(False)
        stats_list = []

        with torch.no_grad():
            for data in tqdm(self.data_loaders['test']):
                img, target = data[0].to(device), data[1].to(device).long()
                self.method.anomaly_score(img)
                discrepancy = 0
                for i, layer in enumerate(self.method.model.modules()):
                    if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.LayerNorm):
                        mu_x, var_x = layer.mean, layer.var
                        mu, var = layer.running_mean, layer.running_var
                        # Calculate KL divergence
                        discrepancy += 0.5 * (torch.log((var + epsilon) / (var_x + epsilon)) +
                                              (var_x + (mu_x - mu) ** 2) / (var + epsilon) - 1).sum()
                stats_list.append(discrepancy.item())

        stats = np.array(stats_list)

        print(f'Saving stats/{args.dataset}_{opt.model.backbone}_{opt.model.method}_stats.npy')
        np.save(f'stats/{args.dataset}_{opt.model.backbone}_{opt.model.method}_stats.npy', stats)

        return

    """
    Functions related to logging (for debug usage).
    """

    def log_init(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        logfile = opt.log_dir + "/log.txt"
        fh = logging.FileHandler(logfile)#, mode='w') # whether to clean previous file
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.handlers = [fh, ch]

        logging.info(str(opt))
        logging.info('Time: %s' % datetime.datetime.now())

    def log_epoch(self, data, epoch, phase='train'):
        phrase = '{} Epoch: {} '.format(phase, epoch)
        for key, value in data.items():
            phrase = phrase + '{}: {:.4f} '.format(key, value)
        logging.warning(phrase)

    def calculate_metrcis(self, target, anomaly_npy, id):
        if (target == 1).sum() > 0:
            roc_auc, prc_auc, fpr95 = eval_ood_measure(anomaly_npy, target, train_id_out=1)
            running_terms = {'roc_auc': roc_auc, 'prc_auc': prc_auc, 'fpr95': fpr95}
            self.log_epoch(running_terms, id)
        else:
            self.logger.info("The image contains no outliers.")
        return
