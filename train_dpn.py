from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import msda
from torch.autograd import Variable
from model.model_dpn import DomainPredictor
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy

# Training settings
parser = argparse.ArgumentParser(description='DPN clustering train')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--kl_wt', type=float, default=0.01,
                    help='clustering weight')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_classes_per_batch', type=int, default=2,
                    help='num classes in a batch after sampling')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--num_domains', type=int, default=4)
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--target', type=str, default='ipqrsc',
                    help='last domain is the target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

args = parser.parse_args()
print('Dataset %s Target %s Network %s' %
      (args.dataset, args.target, args.net))
source_loader, target_loader, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_target_%s_clustering' %
                           (args.method, args.net, args.target))

torch.cuda.manual_seed(args.seed)
G = DomainPredictor(args.num_domains)

params_feature_extractor = list(G.dp_model.layer1.parameters()) + list(G.dp_model.layer2.parameters()) \
    + list(G.dp_model.layer3.parameters()) + list(G.dp_model.layer4.parameters())
params_classifier = list(G.fc5.parameters()) + list(G.bn_fc5.parameters()) + list(G.dp_layer.parameters())


lr = args.lr
G.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def entropy_loss(self, output):
    criterion = HLoss().cuda()
    return criterion(output)

def get_domain_entropy(self, domain_probs):
    bs, num_domains = domain_probs.size()
    domain_prob_sum = domain_probs.sum(0)/bs
    mask = domain_prob_sum.ge(0.000001)
    domain_prob_sum = domain_prob_sum*mask + (1-mask.int())*1e-5
    return -(domain_prob_sum*(domain_prob_sum.log())).mean()

# TODO: take care of Learning rates for G and F.
def train():
    G.train()
    optimizer_g = optim.SGD(params_feature_extractor, lr = 0.1, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(params_classifier, lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    all_step = args.steps
    data_iter_s = iter(source_loader)
    len_train_source = len(source_loader)
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_s = next(data_iter_s)
        im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        zero_grad_all()

        data = im_data_s
        target = gt_labels_s
        domain_logits = G(data)
        domain_prob_s = torch.zeros(domain_logits.shape,dtype=torch.float32).cuda()

        kl_loss = 0
        entropy_loss = 0
        num_active_classes = 0
        for i in range(len(class_list)):
            indexes = target == i
            if indexes.sum()==0:
                continue
            entropy_loss_cl, domain_prob_s_cl = entropy_loss(domain_logits[indexes])
            kl_loss += -get_domain_entropy(domain_prob_s_cl)
            entropy_loss += entropy_loss_cl
            domain_prob_s[indexes] = domain_prob_s_cl
            num_active_classes+=1

        entropy_loss = entropy_loss * args.kl_wt / num_active_classes
        kl_loss = kl_loss * args.kl_wt / num_active_classes

        loss = entropy_loss + kl_loss
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        log_train = 'Target {} Train Ep: {} lr{} \t ' \
                        'Total Loss Clustering: {:.6f} KL Loss: {:.6f} Entropy Loss: {:.6f} Method {}\n'.\
                format(args.target,
                       step, lr, loss.data, kl_loss.data, entropy_loss.data,
                       args.method)



        G.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            G.train()
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('Target {} Train Ep: {} lr{} \t ' \
                        'Total Loss Clustering: {:.6f} KL Loss: {:.6f} Entropy Loss: {:.6f} Method {}\n'.\
                format(args.target,
                       step, lr, loss.data, kl_loss.data, entropy_loss.data,
                       args.method))

            G.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_clustering_model_{}_"
                                        "target_{}_step_{}.pth.tar".
                                        format(args.method, args.target, step)))

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        domain_prob = F.softmax(x, dim=1)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b, domain_prob + 1e-5

train()
