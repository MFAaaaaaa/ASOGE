import math
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import transforms

from model_utils import Classifier, generator_fea_deconv, contrastor, LinearAverage
from dataset import officeDataset_target
# from test import val_pclass
from utils import log, entropy_loss, infoNCE, infoNCE_g, gentropy_loss, CrossEntropyLabelSmooth
import time
from scipy.spatial.distance import cdist
from tensorboardX import SummaryWriter
from auto_augment import AutoAugment


######################
# params             #
######################

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.loss = nn.CrossEntropyLoss().cuda()
        self.loss_entropy = entropy_loss().cuda()
        self.gloss_entropy = gentropy_loss().cuda()
        self.infonce = infoNCE(class_num=31)
        self.gen_c = infoNCE_g(class_num=31)
        self.writer = SummaryWriter()
        self.alpha = 1
        self.logger = log()
        self.lr = args.lr
        self.same_ind = np.array([])
        self.confi_pre = np.array([])

    def cosine_similarity(self, feature, pairs):
        feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
        pairs = F.normalize(pairs)
        similarity = feature.mm(pairs.t())  # 计算余弦相似度
        return similarity  # 返回余弦相似度

    def exp_lr_scheduler(self, optimizer, init_lr, cur_epoch, args):
        """Decay the learning rate based on schedule"""
        cur_lr = init_lr * 0.05 * (1. + math.cos(math.pi * cur_epoch / args.max_epoch))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr
                
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size_m = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size_m).cuda()
        else:
            index = torch.randperm(batch_size_m)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def val_pclass(self, net, test_loader):
        net.eval()
        start_test = True
        with torch.no_grad():
            iter_test = iter(test_loader)
            for i in range(len(test_loader)):
                data = iter_test.__next__()
                inputs = data[0]
                labels = data[1]
                # office31需要下面两行
                labels = np.array(labels).astype(int)
                labels = torch.from_numpy(labels)

                inputs = inputs.cuda()
                outputs, _ = net(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        mean_ent = torch.mean(self.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
        # matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        # acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        # aacc = acc.mean()
        # aa = [str(np.round(i, 2)) for i in acc]
        # acc = ' '.join(aa)
        return accuracy * 100, mean_ent

    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')

        path = self.args.data_path
        source_path = self.args.source_model_path
        label_file = self.args.label_path
        self.logger.info('source_model: ' + source_path.split('/')[-1])
        time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
        self.logger.info(path.split('/')[-2] + time_stamp_launch)
        best_acc = 0
        model_root = './model_' + path.split('/')[-2]

        if not os.path.exists(model_root):
            os.mkdir(model_root)
        cuda = True
        cudnn.benchmark = True
        batch_size = self.args.batchsize
        batch_size_t = int(batch_size / 2)
        batch_size_g = 2 * batch_size
        image_size = (224, 224)
        num_cls = self.args.num_class

        n_epoch = self.args.max_epoch
        weight_decay = 1e-6
        momentum = 0.9

        manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        #######################
        # load data           #
        #######################
        target_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])

        dataset_train = officeDataset_target(path, label_file, train=True, transform=target_train)  # 996

        dataloader_train = torch.utils.data.DataLoader(  # 8
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3
        )
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])

        test_dataset = officeDataset_target(path, label_file, train=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=3)

        #####################
        #  load model       #
        #####################
        self.lemniscate = LinearAverage(2048, test_dataset.__len__(), 0.05, 0.00).cuda()
        # self.elr_loss = elr_loss(num_examp=test_dataset.__len__(), num_classes=31).cuda()

        generator = generator_fea_deconv(class_num=num_cls)

        source_net = torch.load(self.args.source_model_path)
        source_classifier = Classifier(num_classes=num_cls)
        fea_contrastor = contrastor()

        # load pre-trained source classifier
        fc_dict = source_classifier.state_dict()
        pre_dict = source_net.state_dict()
        pre_dict = {k: v for k, v in pre_dict.items() if k in fc_dict}
        fc_dict.update(pre_dict)
        source_classifier.load_state_dict(fc_dict)

        # generator = DataParallel(generator, device_ids=[0, 1])
        # fea_contrastor = DataParallel(fea_contrastor, device_ids=[0, 1])
        # source_net = DataParallel(source_net, device_ids=[0, 1])
        # source_classifier = DataParallel(source_classifier, device_ids=[0, 1])
        source_classifier.eval()

        for p in generator.parameters():
            p.requires_grad = True
        for p in source_net.parameters():
            p.requires_grad = True

        # freezing the source classifier
        for name, value in source_net.named_parameters():
            if name[:9] == 'module.fc':
                value.requires_grad = False

        # setup optimizer
        params = filter(lambda p: p.requires_grad, source_net.parameters())

        model_params = []
        for v in params:
            model_params += [{'params': v, 'lr': self.lr}]

        contrastor_para = []
        for k, v in fea_contrastor.named_parameters():
            contrastor_para += [{'params': v, 'lr': self.lr * 5}]

        #####################
        # setup optimizer   #
        #####################

        # only train the extractor
        optimizer = optim.SGD(model_params + contrastor_para, momentum=momentum,
                              weight_decay=weight_decay)
        optimizer_g = optim.SGD(generator.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)

        loss_gen_ce = torch.nn.CrossEntropyLoss()
        if cuda:
            source_net = source_net.cuda()
            generator = generator.cuda()
            fea_contrastor = fea_contrastor.cuda()
            loss_gen_ce = loss_gen_ce.cuda()
            source_classifier = source_classifier.cuda()

        #############################
        # training network          #
        #############################

        len_dataloader = len(dataloader_train)  # 8
        self.logger.info('the step of one epoch: ' + str(len_dataloader))

        mem_fea = torch.rand(len(dataset_train), 2048).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(len(dataset_train), self.args.num_class).cuda() / self.args.num_class

        current_step = 0
        # rho = 0.9
        half = n_epoch // 2
        for epoch in range(n_epoch):

            generator.train()
            source_net.train()
            fea_contrastor.train()

            data_train_iter = iter(dataloader_train)  # 8  ， 8
            i = 0
            while i < len_dataloader:  # 8
                p = float(i + (epoch) * len_dataloader) / (100) / len_dataloader
                self.p = 2. / (1. + np.exp(-10 * p)) - 1
                # 加载target feature
                data_target_train = data_train_iter.next()  # 8
                t_img, t_label, t_indx = data_target_train
                t_label = np.array(t_label).astype(int)
                t_label = torch.from_numpy(t_label)  # 64

                if cuda:
                    t_img = t_img.cuda()
                    t_label = t_label.cuda()

                toutputs, tfeatures = source_net(t_img)  
                reflect_fea = fea_contrastor(tfeatures)

                dis = -torch.mm(tfeatures.detach(), mem_fea.t())  
                for di in range(dis.size(0)):
                    dis[di, t_indx[di]] = torch.max(dis)
                _, p1 = torch.sort(dis, dim=1)

                w = torch.zeros(tfeatures.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(5):
                        w[wi][p1[wi, wj]] = 0.2
                
                weight_, ppred = torch.max(w.mm(mem_cls), 1)  # tensor 64

                #####################
                # generation    #####
                #####################

                # train generator
                z = Variable(torch.rand(t_img.size(0), 100)).cuda()
                labels = Variable(torch.randint(0, num_cls, (t_img.size(0),))).cuda()
                z = z.contiguous()
                labels = labels.contiguous()
                images_g = generator(z, labels)
                output_teacher_batch = source_classifier(images_g)
                images_g_ref = fea_contrastor(images_g)
                # One hot loss
                loss_one_hot = loss_gen_ce(output_teacher_batch, labels)


                # contrastive loss
                total_contrastive_loss = torch.tensor(0.).cuda()
                contrastive_label = torch.tensor([0]).cuda()
                # NCE
                gamma = 1
                nll = nn.NLLLoss()
    
                for idx in range(images_g.size(0)):
                    pairs4q = self.gen_c.get_posAndneg(features=reflect_features, labels=ppred, tgt_label=labels,
                                                           feature_q_idx=idx, co_fea=images_g_ref[idx].cuda())
                    result = self.cosine_similarity(images_g_ref[idx].unsqueeze(0), pairs4q)

                    numerator = torch.exp((result[0][0]) / gamma)
                    denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                    # log
                    result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                    # nll_loss
                    contrastive_loss = nll(result, contrastive_label)
                    total_contrastive_loss = total_contrastive_loss + contrastive_loss
                total_contrastive_loss = total_contrastive_loss / images_g.size(0)

                # loss of Generator
                optimizer_g.zero_grad()
                loss_G = loss_one_hot - total_contrastive_loss

                loss_G.backward(retain_graph=True)
                self.exp_lr_scheduler(optimizer=optimizer_g, init_lr=self.lr, cur_epoch=epoch, args=self.args)
                optimizer_g.step()

                # adaptation
                # learning rate decay
                optimizer.zero_grad()

                generator.eval()
                images_ad = generator(z, labels)

                all_sam_indx, all_in, _ = np.intersect1d(t_indx, t_indx, return_indices=True)

                img_ad_con = fea_contrastor(images_ad)

                total_contrastive_loss_ad = Variable(torch.tensor(0.).cuda())
                contrastive_label_ad = torch.tensor([0]).cuda()

                # MarginNCE
                gamma = 0.07
                nll_ad = nn.NLLLoss()
                if len(all_in) > 0:
                    for idx in range(len(all_in)):
                        pairs4q = self.infonce.get_posAndneg(features=img_ad_con, labels=labels,
                                                             tgt_label=ppred,
                                                             feature_q_idx=all_in[idx],
                                                             co_fea=reflect_fea[all_in[idx]].cuda())

                        # calculate cosine similarity [-1 1]
                        result = self.cosine_similarity(reflect_fea[all_in[idx]].unsqueeze(0).cuda(), pairs4q)

                        # MarginNCE
                        # softmax
                        numerator = torch.exp((result[0][0]) / gamma)
                        denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                        # log
                        result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                        # nll_loss
                        contrastive_loss_ad = nll_ad(result, contrastive_label_ad) * weight_[all_in[idx]]
                        total_contrastive_loss_ad = total_contrastive_loss_ad + contrastive_loss_ad
                    total_contrastive_loss_ad = total_contrastive_loss_ad / len(all_in)

                tfeatures_m, ppred_a, ppred_b, lam = self.mixup_data(tfeatures, ppred, self.alpha, True)
                tfeatures_m, ppred_a, ppred_b = map(Variable, (tfeatures, ppred_a, ppred_b))
                loss_ce_ad = 0.9 * (lam * loss_gen_ce(toutputs, ppred_a) + (1 - lam) * loss_gen_ce(
                        toutputs, ppred_b))
                
                loss = total_contrastive_loss_ad + loss_ce_ad
                loss.backward()
                self.exp_lr_scheduler(optimizer=optimizer, init_lr=self.lr, cur_epoch=epoch, args=self.args)
                optimizer.step()

                if epoch <= 3:
                    source_net.eval()
                    with torch.no_grad():
                        output_t, fea_t = source_net(t_img)
                        fea_t = fea_t / torch.norm(fea_t, p=2, dim=1, keepdim=True)
                        softmax_out = nn.Softmax(dim=1)(output_t)
                        output_t = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))
                    mem_fea[t_indx] = fea_t.clone()
                    mem_cls[t_indx] = output_t.clone()

                i += 1
                current_step += 1

            self.logger.info('epoch: %d' % epoch)

            self.logger.info('loss_G: %f' % loss_G)
            self.logger.info('loss: %f' % loss)
            accu, ac_list = self.val_pclass(source_net, test_loader)

            if accu >= best_acc:
                self.logger.info('saving the best model!')
                torch.save(source_net,
                           model_root + '/' + time_stamp_launch + '_best_model_' + path.split('/')[-2] + '.pkl')
                best_acc = accu

            self.logger.info('acc is : %.04f, best acc is : %.04f' % (accu, best_acc))
            self.logger.info('================================================')

        self.logger.info('training done! ! !')
