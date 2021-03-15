import os
import torch
import torch.nn as nn
import shutil
import random
from torch.utils.data.sampler import Sampler
import itertools

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

class SubsetClassRandomSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes_per_batch, num_classes):
        assert batch_size%num_classes_per_batch==0
        self.num_classes = num_classes
        self.labels = labels[:]
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        label_map = {}
        for i in range(num_classes):
            label_map[i] = []
        for idx, label in enumerate(labels):
            label_map[label].append(idx)
        self.label_map = label_map
        self.total_batches = len(labels)*num_classes_per_batch//batch_size
        self.num_samples_per_class_per_batch = batch_size//num_classes

    def __iter__(self):
        class_samples = {}
        for i in range(self.num_classes):
            class_samples[i] = random.sample(self.label_map[i],
                                             self.num_samples_per_class_per_batch*self.total_batches)
        all_combos = itertools.combinations(list(range(self.num_classes)),self.num_clsses_per_batch)
        sampled_combos = random.choices(all_combos, self.total_batches)
        batches = []
        for batch_idx, label_group in enumerate(sampled_combos):
            cur_batch = []
            for label in label_group:
                cur_batch.extend(class_samples[label][batch_idx*(self.num_samples_per_class_per_batch)
                                             :(batch_idx+1)*self.num_samples_per_class_per_batch])
            batches.append(cur_batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.total_batches