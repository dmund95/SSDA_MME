import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist
from utils import SubsetClassRandomSampler
import random
from torch.utils.data.sampler import Sampler
import itertools

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

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
            class_samples[i] = random.choices(self.label_map[i],
                                             k=self.num_samples_per_class_per_batch*self.total_batches)
        all_combos = list(itertools.combinations(list(range(self.num_classes)),self.num_classes_per_batch))
        sampled_combos = random.choices(all_combos, k=self.total_batches)
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

def return_dataset(args):
    base_path = '/vulcan-pvc1/data/domain_net/'

    domain_map = {'i':'infograph', 'p':'painting', 'q':'quickdraw', 'r':'real', 's':'sketch', 'c':'clipart'}

    image_set_file_s = []

    target_domain = domain_map[args.target[-1]]
    source_domains = [domain_map[args.target[i]] for i in range(len(args.target)-1)]

    for sd in source_domains:
        image_set_file_s.append(os.path.join(base_path,sd+'_train.txt'))
    
    image_set_file_t = [os.path.join(base_path,target_domain+'_train.txt')]

    image_set_file_t_val = [os.path.join(base_path,target_domain+'_test.txt')]
    image_set_file_unl = [os.path.join(base_path,target_domain+'_test.txt')]

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s[0])
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    
    sampler = SubsetClassRandomSampler(source_dataset.labels, bs, args.num_classes_per_batch, len(class_list))

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_sampler=sampler, num_workers=3)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, \
        target_loader_val, target_loader_test, class_list


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
