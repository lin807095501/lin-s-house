from torchvision import transforms
import torchvision
from .TinyImageNet import TinyImageNet
from .autoaugment import CIFAR10Policy,CutoutDefault

def getData(dataset):

    if dataset == 'CIFAR10':

        DATAROOT = '/data/lsb/dataset/cifar10'
        # transform_train = transforms.Compose(
        #                 [transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
        #                  transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
		# 	             transforms.ToTensor()])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_data = torchvision.datasets.CIFAR10(
            root=DATAROOT, train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.CIFAR10(
            root=DATAROOT, train=False, download=True, transform=transform_test)

        num_classes = 10

        return num_classes, train_data, test_data

    if dataset == 'CIFAR100':
        DATAROOT = '/data/lsb/dataset/cifar100'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_data = torchvision.datasets.CIFAR100(
            root=DATAROOT, train=True, download=False, transform=transform_train)
        test_data = torchvision.datasets.CIFAR100(
            root=DATAROOT, train=False, download=False, transform=transform_test)
        num_classes = 100

        return num_classes, train_data, test_data

    if dataset == 'Tiny_Image':

        DATAROOT = '/data/datasets/Tiny_Imagenet/tiny-imagenet-200/'
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
        ])

        in_memory = False
        train_data = TinyImageNet(
            root=DATAROOT,
            split='train',
            transform=transform_train,
            in_memory=in_memory)
        test_data = TinyImageNet(
            root=DATAROOT,
            split='val',
            transform=transform_test,
            in_memory=in_memory)

        num_classes = 200

        return num_classes, train_data, test_data



