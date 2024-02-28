import torchvision
import umap
import umap.plot
import timm
import torch
import torch.utils.data


def load_data(dataset_path, train=True):
    images_data = torchvision.datasets.FashionMNIST(root=dataset_path, train=train, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    # data_loader = torch.utils.data.DataLoader(data,
    #                                           batch_size=4,
    #                                           shuffle=False,
    #                                           num_workers=2)

    return images_data.data, images_data.targets, images_data.classes


dataset_path = '/media/amidemo/Data/Fashion-MNIST'
#
X_train, y_train, class_names = load_data(dataset_path, train=True)
# X_test, y_test, class_names = load_data(dataset_path, train=False)
#
# mapper = umap.UMAP().fit(X_train.view(X_train.shape[0], -1))
# p = umap.plot.points(mapper, labels=y_train)
#
# for cn, text in zip(class_names, p.legend_.get_texts()):
#     text.set_text(cn)
#
# umap.plot.show(p)

#avail_pretrained_models = timm.list_models(pretrained=True)
pass