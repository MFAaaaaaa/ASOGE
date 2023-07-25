# coding=utf-8
"""
validation
"""
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import manifold, datasets
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix
import torch.utils.data
import argparse
import network
from torch import nn

from dataset import visDataset_target, officeDataset_target


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def val_source(net, test_loader):
    net.eval()
    correct = 0
    total = 0

    gt_list = []
    p_list = []

    for i, (inputs, labels, _) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        gt_list.append(labels.cpu().numpy())
        with torch.no_grad():
            outputs, _ = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        output_prob = F.softmax(outputs, dim=1).data
        p_list.append(output_prob[:, 1].detach().cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

    acc = 100. * correct.item() / total
    prob_list = np.concatenate(p_list)
    gt_list = np.concatenate(gt_list)

    return acc


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]

            labels = np.array(labels).astype(int)
            labels = torch.from_numpy(labels)

            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            fea = netB(netF(inputs))
            if start_test:
                all_fea = fea.cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_fea = torch.cat((all_fea, fea.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)

    tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=900, perplexity=60)
    result = tsne.fit_transform(all_fea)
    # ys = matplotlib.colors.ListedColormap(
    #     ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
    cm1 = plt.get_cmap('Paired')
    # cm2 = plt.get_cmap('Pastel1')
    # cm3 = plt.get_cmap('Pastel2')
    # ys = ListedColormap(cm1.colors + cm2.colors + cm3.colors)
    ys = ListedColormap(cm1.colors)
    plt.scatter(result[:, 0], result[:, 1], c=predict, marker='.', cmap=ys)
    plt.savefig('visualiaztion-shot.jpg')

    return aacc, acc


def val_pclass(net, test_loader):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = iter_test.next()
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
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()


    # matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    # acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    # aacc = acc.mean()
    # aa = [str(np.round(i, 2)) for i in acc]
    # acc = ' '.join(aa)
    # return aacc, acc
    return accuracy * 100, mean_ent


def vis_pclass(net, test_loader):
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
            outputs, feas = net(inputs)
            if start_test:
                all_feas = feas.cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_feas = torch.cat((all_feas, feas.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    # print(predict.size(), all_label.size())

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    # return accuracy * 100, mean_ent

    # matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    # acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    # aacc = acc.mean()
    # aa = [str(np.round(i, 2)) for i in acc]
    # acc = ' '.join(aa)
    # implot = plt.imshow(im)
    # tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=100, n_iter=1500, perplexity=35) # source
    tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=300, n_iter=1500, perplexity=35)
    # tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=900, perplexity=60)
    result = tsne.fit_transform(all_feas)
    # ys = matplotlib.colors.ListedColormap(
    #     ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
    ys = matplotlib.colors.ListedColormap('b')
    # cm1 = plt.get_cmap('Paired')
    # cm2 = plt.get_cmap('Pastel1')
    # cm3 = plt.get_cmap('Pastel2')
    # ys = ListedColormap(cm1.colors + cm2.colors + cm3.colors)
    # ys = ListedColormap(cm1.colors)
    plt.scatter(result[:, 0], result[:, 1], s=40, c=predict, marker='.', cmap=ys, )
    plt.savefig('st.png')
    # return aacc, acc
    return accuracy * 100, mean_ent


def st_pclass(net, net2, test_loader1, test_loader2):
    net.eval()
    net2.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(test_loader1)
        for i in range(len(test_loader1)):
            data = iter_test.__next__()
            inputs = data[0]
            labels = data[1]
            # office31需要下面两行
            labels = np.array(labels).astype(int)
            labels = torch.from_numpy(labels)

            inputs = inputs.cuda()
            outputs, feas = net(inputs)
            if start_test:
                all_feas = feas.cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_feas = torch.cat((all_feas, feas.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    # print(predict.size(), all_label.size())

    start_test2 = True
    with torch.no_grad():
        iter_test2 = iter(test_loader2)
        for i2 in range(len(test_loader2)):
            data2 = iter_test2.__next__()
            inputs2 = data2[0]
            labels2 = data2[1]
            # office31需要下面两行
            labels2 = np.array(labels2).astype(int)
            labels2 = torch.from_numpy(labels2)

            inputs2 = inputs2.cuda()
            outputs2, feas2 = net(inputs2)
            if start_test2:
                all_feas2 = feas2.cpu()
                all_output2 = outputs2.float().cpu()
                all_label2 = labels2.float()
                start_test2 = False
            else:
                all_output2 = torch.cat((all_output2, outputs2.float().cpu()), 0)
                all_label2 = torch.cat((all_label2, labels2.float()), 0)
                all_feas2 = torch.cat((all_feas2, feas2.cpu()), 0)
    _2, predict2 = torch.max(all_output2, 1)

    # return accuracy * 100, mean_ent

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    # tsne = manifold.TSNE(n_components=2, init='pca', learning_rate=300, perplexity=40)
    tsne = manifold.TSNE(n_components=2, learning_rate=100, init='pca', n_iter=700, perplexity=30,
                         early_exaggeration=12)# 48
    # tsne2 = manifold.TSNE(n_components=2, learning_rate=50, init='pca', n_iter=2000, perplexity=35,
    #                      early_exaggeration=15)# 48
    tsne2 = manifold.TSNE(n_components=2, learning_rate=50, init='pca', n_iter=420, perplexity=35,
                         early_exaggeration=12)# 48

    result = tsne.fit_transform(all_feas)
    result2 = tsne2.fit_transform(all_feas2)
    ys1 = matplotlib.colors.ListedColormap('r')
    ys2 = matplotlib.colors.ListedColormap('b')
    # ys1 = matplotlib.colors.ListedColormap(
    #     ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
    # cm1 = plt.get_cmap('Paired')
    # cm2 = plt.get_cmap('Pastel1')
    # cm3 = plt.get_cmap('Pastel2')
    # ys = ListedColormap(cm1.colors + cm2.colors + cm3.colors)
    # ys = ListedColormap(cm1.colors)
    plt.scatter(result[:, 0], result[:, 1], s=40, c=predict, marker='.', cmap=ys1, )
    plt.scatter(result2[:, 0], result2[:, 1], s=40, c=predict2, marker='.', cmap=ys2, )

    plt.savefig('ss.png')
    return aacc, acc
    # return accuracy * 100, mean_ent


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='2', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=128, type=int)
    # parser.add_argument('--test_path',
    #                     default='/home/fameng/pyCharmProject/TCPGA/model_VISDA-C/20230214-1108_best_model_VISDA-C.pkl',
    #                     type=str, help='path to the cpga model')

    parser.add_argument('--test_path2',
                        default='/home/fameng/pyCharmProject/TCPGA/model_VISDA-C/20230114-0928_best_model_VISDA-C.pkl',
                        type=str, help='path to the ad model')
    parser.add_argument('--test_path',default='/home/fameng/pyCharmProject/TCPGA/model_source/20220405-1028-VISDA-C9_1_resnet100_best.pkl',
                        type=str,help='path to the pre-trained source model')
    # parser.add_argument('--data_path', default='/home/fameng/pyCharmProject/TCPGA/dataset/VISDA-C/validation', type=str,
    #                     help='path to source data')
    # parser.add_argument('--test_path',
    #                     default='/home/fameng/pyCharmProject/TCPGA/model_Clipart/20221212-1721_best_model_Clipart.pkl',
    #                     type=str, help='path to the ad model')
    # parser.add_argument('--data_path', default='/home/fameng/pyCharmProject/TCPGA/dataset/VISDA-C/visualTess', type=str,
    #                     help='path to target data')
    parser.add_argument('--data_path', default='/home/fameng/pyCharmProject/TCPGA/dataset/VISDA-C/source', type=str,
                        help='path to target data')
    parser.add_argument('--data_path2', default='/home/fameng/pyCharmProject/TCPGA/dataset/VISDA-C/target', type=str,
                        help='path to target data')

    # parser.add_argument('--label_file', default='./data/visda_real_train.pkl', type=str)
    parser.add_argument('--label_file', default='./data_utils/strain.pkl', type=str)
    parser.add_argument('--label_file2', default='./data_utils/ttrain.pkl', type=str)

    # parser.add_argument('--netB_path',
    #                     default='/home/fameng/pyCharmProject/SHOT/object/ckps/target/uda/VISDA-C/TV/target_B_par.pt',
    #                     type=str, help='path to netB')
    # parser.add_argument('--netF_path',
    #                     default='/home/fameng/pyCharmProject/SHOT/object/ckps/target/uda/VISDA-C/TV/target_F_par.pt',
    #                     type=str, help='path to netF')
    # parser.add_argument('--netC_path',
    #                     default='/home/fameng/pyCharmProject/SHOT/object/ckps/target/uda/VISDA-C/TV/target_C_par.pt',
    #                     type=str, help='path to netC')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    net = torch.load(args.test_path).cuda()
    net2 = torch.load(args.test_path2).cuda()

    # net = net.module
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
    ])

    val_dataset = officeDataset_target(args.data_path, args.label_file, train=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False,
                                             num_workers=3)

    val_dataset2 = officeDataset_target(args.data_path2, args.label_file2, train=False, transform=transform_test)
    val_loader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=args.batchsize, shuffle=False,
                                              num_workers=3)

    # acc = vis_pclass(net, val_loader2)
    acc = st_pclass(net, net2, val_loader, val_loader2)
    # acc = cal_acc(val_loader, netF, netB, netC, flag=False)
    print(acc)
