"""
Author: Nazanin Moradinasab

Universal representation learning for multivariate time series using the instance‐level and cluster‐level supervised contrastive learning

"""
# Import libraries
import pandas as pd
from utils import *
from models.imports import *
from layers import *
from layers import ConvBlock
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
import torch.nn.functional as F
from losses_new import SupConLoss
import random
import os
from dataset import Select_Dataset
import time
import sys
import argparse
from sklearn.preprocessing import LabelEncoder

__all__ = ['ResBlock', 'ResNet']


def main(x_train_np, x_test_np, y_train_np, y_test_np, name_dataset, list_seeds,num_classes):
    time1 = time.time()

    # Reproducibility
    for seed in list_seeds:
        # For reproducibility
        init_reproducible(seed=seed)

        # set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare labels
        label_Encoder = LabelEncoder()
        y_train_np_numerical = label_Encoder.fit_transform(y_train_np)

        #split train set to n=n_fold folds (to perform Ensembling)
        train_indx, test_indx = data_spliter(x_train_np, y_train_np_numerical, n=n_fold)

        # Inintialization
        EPOCHS = args.Epoch
        batch_size_emb = args.batch_size_emb  # batch size value for pre-training using the contrastive learning
        batch_size_cl = args.batch_size_cl  # batch size value for training the classifier
        temperature = 0.07
        base_temperature = 0.07

        # train transform (for both target network and source network)
        train_transform_source = transforms.Compose([Jittering(sigma=0.0010), transforms.ToTensor()])
        train_transform_target = transforms.Compose([Jittering(sigma=0.0010), transforms.ToTensor()])  # 1
        
        # train transform
        val_transform = transforms.Compose([transforms.ToTensor()])
        
        # putting transforms together for source and target network
        transform = TwoCropTransform(train_transform_source, train_transform_target)
        
        average_accuracy_val = []
        average_accuracy_test = []
        
        for i in range(n_fold):

            # prepare datasets and dataloaders
            train_set_whole = Dataset(x_train_np, y_train_np, transform)   # whole trainset
            train_set = Dataset(x_train_np[train_indx[i]], y_train_np[train_indx[i]], transform) # train split from train-validation splits
            test_set = Dataset(x_test_np, y_test_np, val_transform)  # whole testset
            val_set = Dataset(x_train_np[test_indx[i]], y_train_np[test_indx[i]], val_transform) # validation split from train-validation splits
            train_set.__getitem__(0)

            # creat train and test loader
            train_loader_whole_emb = torch.utils.data.DataLoader(train_set_whole, batch_size=batch_size_emb, shuffle=True)
            train_loader_whole_cl = torch.utils.data.DataLoader(train_set_whole, batch_size=batch_size_cl, shuffle=True)
            train_loader_emb = torch.utils.data.DataLoader(train_set, batch_size=batch_size_emb, shuffle=True)
            train_loader_cl = torch.utils.data.DataLoader(train_set, batch_size=batch_size_cl, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
            val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

            # build the model, loss, and optimizer
            model = ResNet(c_in=x_train_np.shape[1], c_out=num_classes, buffer_size= batch_size_value)
            model = model.double()
            loss_function_CE = torch.nn.CrossEntropyLoss()  # No softmax at the last layer
            ContLr_loss_function = SupConLoss(contrast_mode='one', temperature=temperature,
                                              base_temperature=base_temperature)
            # set optimizer
            optimizer1 = torch.optim.RMSprop(model.parameters(), lr=args.lr1)
            optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr2)

            # pre-train the model
            model.train()
            pre_train(model, train_loader_emb, ContLr_loss_function, optimizer1, 100, num_classes, device)

            # freeze the encoder
            for name, param in model.named_parameters():
                if 'conv' in name:
                    param.requires_grad = False
                if 'head' in name:
                    param.requires_grad = False

            # train the classifier
            accuracy_test, accuracy_val = train(model, train_loader_cl, val_set, test_loader, loss_function_CE, optimizer2, EPOCHS, device,
                                  name_dataset, i)
            average_accuracy_val.append(accuracy_val)
            average_accuracy_test.append(accuracy_test)

        print(f"The average val accuracy on folds is: {np.mean(average_accuracy_val)}")

        #train on whole dataset
        model = ResNet(c_in=x_train_np.shape[1], c_out=num_classes, buffer_size= batch_size_value)

        model = model.double()
        loss_function_CE = torch.nn.CrossEntropyLoss()  # No softmax at the last layer, no one hot embedding
        ContLr_loss_function = SupConLoss(contrast_mode='one', temperature=temperature,
                                          base_temperature=base_temperature)
        # set optimizer
        optimizer1 = torch.optim.RMSprop(model.parameters(), lr=args.lr1)  # 5e-6
        optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr2)  # 5e-5

        # pre-train the model
        model.train()
        pre_train(model, train_loader_whole_emb, ContLr_loss_function, optimizer1, 10, num_classes, device)

        for name, param in model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False
            if 'head' in name:
                param.requires_grad = False

        # train the classifier
        accuracy_test, accuracy_val = train(model, train_loader_whole_cl, val_set, test_loader, loss_function_CE, optimizer2, EPOCHS, device,
                                  name_dataset, "seed")
        print(f"The average test accuracy (train on whole dataset) on folds is: {accuracy_test}")

    ensemble_results(model, test_loader, loss_function_CE, device, name_dataset,
                     list_seeds, num_classes)   #Add to final
    time1 = time.time() - time1
    print('time:', time1)

########################################################################################
# grad cam
def grad_cam(model, test_set, device, name_dataset,seed):
    """ Generate the Grad cam for the given sample"""
    import cv2
    model.load_state_dict(torch.load('./data/'+ name_dataset + '/best_en_'+ str(seed)))   #Add to final
    for name, param in model.named_parameters():
        if 'conv' in name:
            param.requires_grad = True
    model = model.cpu()
    model.eval()

    # sample from test_loader
    random_ids = np.random.randint(len(test_set), size=20)

    for i in random_ids:
        data = test_set[i]

        logit, y_pred = torch.max(model(data["x"], grad='True')[0], 1)
        # get the gradient of the output with respect to the parameters of the model
        logit.backward()
        # pull the gradients out of the model
        gradients = model.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2])
        # get the activations of the last convolutional layer
        activations = model.get_activations(data["x"].squeeze(1)).detach()
        # weight the channels by corresponding gradients
        for c in range(128):
            activations[:, c, :] *= pooled_gradients[c]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)
        # normalize the heatmap
        heatmap /= (torch.max(heatmap) + 0.0000000000001)

        heatmap = heatmap.unsqueeze(0).numpy()
        h, w = data["x"].squeeze(1).squeeze(0).shape
        heatmap_resize = cv2.resize(heatmap, (w, h))

        plt.matshow(heatmap_resize)
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            right=False,  # ticks along the right edge are off
            left=False, labelleft=False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
        )
        plt.savefig('./data/'+name_dataset + '/gradcam/sample_' + str(
            i) + '_label_' + str(data["y"].item()) + '_time.png')
#         plt.show()
        # draw heatmap
        for d in range(h):
            plt.scatter(x=range(w), y=data["x"].squeeze(1).squeeze(0)[d, :], c=heatmap_resize[d, :])
            plt.savefig('./data/'+name_dataset + '/gradcam/sample_' + str(
                i) + '_dim_' + str(d) + '.png')
#             plt.show()



##################################### Data splitter ##############################################

def data_spliter(data, targets, n=5):
    """ split the dataset into the train and validation"""
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=20)
    test_subjects = []
    train_subjects = []
    for train_index, test_index in kfold.split(data, targets.astype(int)):
        train_subjects.append(list(train_index))
        test_subjects.append(test_index)
    return train_subjects, test_subjects


##################################### Transform ##############################################
class Jittering(object):
    """ jittering augmentation"""
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):

        if isinstance(self.sigma, float):
            myNoise = np.random.normal(loc=0, scale=self.sigma, size=sample.shape[1])
            data = sample + myNoise
        return data


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, train_transform_source, train_transform_target):
        self.train_transform_source = train_transform_source
        self.train_transform_target = train_transform_target

    def __call__(self, x):
        return [self.train_transform_source(x), self.train_transform_target(x)]


##################################### ResNet ##############################################
# from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py (tsai github)
# timeseriesAI/tsai is licensed under the
# Apache License 2.0
# tsai licese is located in the LICENSE

class ResBlock(torch.nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        super().__init__()
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet model"""

    def __init__(self, c_in, c_out, buffer_size,feat_dim=128, head = 'mlp'):
        super().__init__()
        nf = 64
        kss=[7, 5, 3]
        self.resblock1_conv = ResBlock(c_in, nf, kss=kss)
        self.resblock2_conv = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3_conv = ResBlock(nf * 2, nf * 2, kss=kss)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(nf * 2, 200), torch.nn.ReLU(),
                                              torch.nn.Linear(200, c_out))
        self.register_buffer('my_buffer_x', F.normalize(torch.randn(buffer_size, feat_dim), dim=1, p=2))
        self.register_buffer('my_buffer_y', torch.zeros(buffer_size, 1))
        if head == 'linear':
            self.head = torch.nn.Linear(128, feat_dim)
        elif head == 'mlp':
            self.head = torch.nn.Sequential(
                torch.nn.Linear(128, feat_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(feat_dim, feat_dim))

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x,grad='false'):
        x = self.resblock1_conv(x)
        x = self.resblock2_conv(x)
        x = self.resblock3_conv(x)


        if grad == 'True':
            h = x.register_hook(self.activations_hook)

        feat = self.squeeze(self.gap(x))
        # classification head
        pre = self.classifier(feat)
        # embedding head
        emb = F.normalize(self.head(feat), dim=1, p=2)

        return pre, emb

        # method for the gradient extraction
    def get_activations_gradient(self):
            return self.gradients

        # method for the activation exctraction
    def get_activations(self, x):
            x = self.resblock1_conv(x)
            x = self.resblock2_conv(x)
            x = self.resblock3_conv(x)
            return x






class InceptionModule(Module):
    """Inception model"""
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([Conv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.bn = BN1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return self.act(self.bn(x))


@delegates(InceptionModule.__init__)
class InceptionBlock(Module):
    def __init__(self, ni, nf=32, residual=True, depth=6, **kwargs):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs))
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, 1, act=None))
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(self.add(x, self.shortcut[d//3](res)))
        return x


@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(self, c_in, c_out, buffer_size, seq_len=None, nf=32, nb_filters=None,feat_dim = 64, head = 'mlp', **kwargs):
        nf = ifnone(nf, nb_filters) # for compatibility
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        self.gap = GAP1d(1)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(nf * 4, 200), torch.nn.ReLU(),
                                              torch.nn.Linear(200, c_out))
        #
        self.register_buffer('my_buffer_x', F.normalize(torch.randn(buffer_size, feat_dim), dim=1, p=2))
        self.register_buffer('my_buffer_y', torch.zeros(buffer_size, 1))
        if head == 'linear':
            self.head = torch.nn.Linear(128, feat_dim)
        elif head == 'mlp':
            self.head = torch.nn.Sequential(
                torch.nn.Linear(128, feat_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(feat_dim, feat_dim))

    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, x,grad= 'False'):
        x = self.inceptionblock(x)


        if grad == 'True':
            h = x.register_hook(self.activations_hook)

        feat = self.gap(x)
        # classification head
        pre = self.classifier(feat.double())
        # embedding head
        emb = F.normalize(self.head(feat.double()), dim=1, p=2)

        return pre, emb

        # method for the gradient extraction

    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        return x

##################################### training the Encoder ##############################################
def pre_train(model, train_loader, loss_function, optimizer, num_epochs, num_classes, device):
    """this function is used to train the encoder part"""
    model = model.to(device)

    ContLr_loss_function = loss_function.to(device)
    loss_plot = []

    for epoch in range(num_epochs):

        model.train()
        alpha = 0
        if epoch > next(iter(train_loader))['x'][0].shape[0] / 2:
            alpha = 1
        for data in train_loader:
            data_x = torch.cat([data["x"][0], data["x"][1]], dim=0).to(device)
            bsz = data["y"].shape[0]

            # compute loss
            y_pred, features = model(data_x.squeeze(1))

            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # f1: source features, f2: target features
            features_source_target = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_CL = ContLr_loss_function(features_source_target, data["y"].to(device))
            # calculating the mean of features per each label and apply F2 normalization on it, then added to the buffer
            for i in range(int(num_classes)):
                if (data["y"] == i).sum() > 0:
                    model.my_buffer_x = torch.cat(
                        [model.my_buffer_x[1:],
                         F.normalize(
                             f2[torch.nonzero((data["y"] == i)).reshape(-1).to(device)].detach().mean(0).unsqueeze(0),
                             dim=1)])
                    model.my_buffer_y = torch.cat(
                        [model.my_buffer_y[1:], torch.tensor(i).unsqueeze(0).unsqueeze(0).to(device)], dim=0)

            # mean of features from previous batches and their corresponding labels
            feature_mean = model.my_buffer_x
            label_mean = model.my_buffer_y
            mask = torch.eq(data["y"].unsqueeze(1).to(device), label_mean.T.to(device)).float()
            if f1.shape[0] != feature_mean.shape[0]:  # TODO: check it
                if feature_mean.shape[0]>f1.shape[0]:
                    sample = np.random.choice(feature_mean.shape[0], f1.shape[0], replace=False)
                else:
                    sample = np.random.choice(feature_mean.shape[0], feature_mean.shape[0], replace=True)
                feature_sample = feature_mean[sample]
                label_sample = label_mean[sample]
                mask = torch.eq(data["y"].unsqueeze(1).to(device), label_sample.T.to(device)).float()

                loss_CL_mean = ContLr_loss_function(torch.cat([f1.unsqueeze(1), feature_sample.unsqueeze(1)], dim=1),
                                                    labels=None,
                                                    mask=mask).to(device)
            else:
                loss_CL_mean = ContLr_loss_function(torch.cat([f1.unsqueeze(1), feature_mean.unsqueeze(1)], dim=1),
                                                    labels=None, mask=mask).to(device)

            loss = (1 - alpha) * loss_CL + alpha * loss_CL_mean

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_plot.append(loss.item())

        plt.plot(loss_plot)
        plt.xlabel("number of epochs")
        plt.ylabel("loss value")


##################################### training the classifier ##############################################

def train(model, train_loader, train_set_test_loader, test_loader, loss_function_CE, optimizer, num_epochs, device, name_dataset, seed):
    """This function is used to train the classifier"""
    model = model.to(device)
    loss_function_CE = loss_function_CE.to(device)

    loss_plot = []
    outputs = []
    accuracy_test = []
    accuracy_val = []
    entropy_weight = 0

    # create a directory to save gradcams
    path = './data/'+  name_dataset + '/gradcam'
    if not os.path.isdir(path):
        os.mkdir(path)
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            model = model.to(device)
            data_x = torch.cat([data["x"][0], data["x"][1]], dim=0).to(device)
            label = torch.cat([data["y"], data["y"]], dim=0).to(device)

            # compute loss
            y_pred, features = model(data_x.squeeze(1).to(device))
            y_pred_prob = F.softmax(y_pred, dim=-1)

            loss_CE = loss_function_CE(y_pred, label.to(device))
            loss_Entropy = xentropy_loss(y_pred_prob, y_pred_prob)

            loss = (1 - entropy_weight) * loss_CE + entropy_weight * loss_Entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_plot.append(loss.item())
        plt.plot(loss_plot)
        plt.xlabel("number of epochs")
        plt.ylabel("loss value")
        # plt.show()
        model.eval()
        accuracy = model_eval(model, test_loader, loss_function_CE, device)
        accuracy_test.append(accuracy)
        if epoch == (num_epochs - 1):
            outputs.append((data["x"], y_pred))
        # save best weights on val set
        accuracy2 = model_eval(model, train_set_test_loader, loss_function_CE, device)
        if best_accuracy < accuracy2:
            torch.save(model.state_dict(),
                       './data/'+ name_dataset + '/best_en_'+str(seed))
            best_accuracy = accuracy2
        accuracy_val.append(accuracy2)
    return accuracy_test[-1], accuracy_val[-1]


##################################### Ensembling ##############################################

def ensemble_results(model, test_loader, loss_function_CE, device, name_dataset,
                     list_seeds,num_classes):
    """This function is used to compute the ensembling results"""
    accuracy_list = []
    Correct_vote = 0
    Correct_sum = 0
    total = 0
    n=0
    for data in test_loader:
        total += 1
        result_ins=[]
        ensemble = torch.zeros((1,num_classes))
        n+=1
        for seed in range(5):
            model.load_state_dict(torch.load("./data/"+ name_dataset + '/best_en_' + str(seed)))
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                test_results = []
                model = model.cpu()
                out =F.softmax(model(data["x"].squeeze(1))[0])
                _, y_pred = torch.max(out, 1)
                ensemble += out
                result_ins.append(y_pred)
        y_true = data["y"]
        # if np.max(result_ins) == y_true:
        #     Correct_vote += 1
        if torch.max(ensemble, 1)[1][0].item() == y_true:
            Correct_sum += 1
    # accuracy_vote = Correct_vote / total
    accuracy_sum = Correct_sum / total

    print('The test accuracy is equal to: Sum: %', accuracy_sum)


##################################### Evaluation ##############################################
def model_eval(model, test_loader, loss_function, device):
    """model evaluation"""
    with torch.no_grad():
        test_results = []
        model = model.cpu()
        Correct = 0
        total = 0
        for data in test_loader:
            total += 1
            _, y_pred = torch.max(model(data["x"].squeeze(1))[0], 1)
            y_true = data["y"]
            if y_pred == y_true:
                Correct += 1
            test_results.append((y_true, y_pred))
    accuracy = Correct / total

    return accuracy


##################################### Reproduciblity ##############################################
def init_reproducible(seed=0):
    """this part is used to ensure the results are reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


##################################### Loss functions ##############################################
# Entropy

def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array

    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "None":
        loss = loss
    else:
        loss = loss.sum()
    return loss

##################################### Normalization ##############################################

def normalize(x_train_np, x_test_np):
    """This function is used to normalize the dataset"""
    x_train = np.moveaxis(x_train_np, 1, 0).reshape((x_train_np.shape[1], -1))
    max_train = np.moveaxis(x_train_np, 1, 0).reshape((x_train_np.shape[1], -1)).max(1)

    x_test = np.moveaxis(x_test_np, 1, 0).reshape((x_test_np.shape[1], -1))
    max_test = np.moveaxis(x_test_np, 1, 0).reshape((x_test_np.shape[1], -1)).max(1)

    max = np.concatenate([max_train[np.newaxis, :], max_test[np.newaxis, :]], 0).max(0)
    x_train_n = x_train / max[:, np.newaxis]
    x_test_n = x_test / max[:, np.newaxis]
    return x_train_n.reshape((x_train_np.shape[0], x_train_np.shape[1], x_train_np.shape[2])), x_test_n.reshape(
        (x_test_np.shape[0], x_test_np.shape[1], x_test_np.shape[2]))


##################################### main function ##############################################
if __name__ == "__main__":
    # Initialization
    batch_size_value = 100
    list_seeds = [0]
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-lr1", "--lr1", type=float, help="Learning rate pre training", default=0.0001)
    parser.add_argument("-lr2", "--lr2", type=float, help="Learning rate classifier", default=0.0001)
    parser.add_argument("-batch_size_emb", "--batch_size_emb", type=int, help="batch_size_emb", default=50)
    parser.add_argument("-batch_size_cl", "--batch_size_cl", type=int, help="batch_size_cl", default=5)
    parser.add_argument("-Epoch", "--Epoch", type=int, help="Epoch", default=50)
    parser.add_argument("-name_dataset", "--name_dataset", help="name_dataset", default="BasicMotions")
    args = parser.parse_args()
    config = vars(args)
    print(config)
    name_dataset = args.name_dataset
    dict_num_classes = {'HandMovementDirection': 4, 'Phoneme': 39, 'ArticularyWordRecognition': 25,
                        'AtrialFibrillation': 3,
                        'BasicMotions': 4, 'CharacterTrajectories': 20, 'FaceDetection': 2, 'Heartbeat': 2, 'NATOPS': 6,
                        'MotorImagery': 2, 'PEMS-SF': 7, 'PenDigits': 10, 'StandWalkJump': 3, 'SelfRegulation_SCP2': 2,
                        'SelfRegulationSCP1': 2, 'Cricket': 12, 'DuckDuckGeese': 5, 'EigenWorms': 5, 'Epilepsy': 4,
                        'EthanolConcentration': 4, 'ERing': 6, 'Handwriting': 26, 'Libras': 15, 'LSST': 14,
                        'RacketSports': 4, 'UWaveGestureLibrary': 8, 'FingerMovements': 2}
    num_classes = dict_num_classes[name_dataset]
    Dataset = Select_Dataset(name_dataset)
    n_fold = 5  # number of folds

    x_train = np.load("./data/"+name_dataset + "/x_train.npy")
    x_train = np.moveaxis(x_train, 1, 2)

    x_test = np.load("./data/"+ name_dataset + "/x_test.npy")
    x_test = np.moveaxis(x_test, 1, 2)

    y_test = np.load("./data/"+name_dataset + "/y_test.npy")
    y_train = np.load("./data/"+ name_dataset + "/y_train.npy")

    main(x_train, x_test, y_train, y_test, name_dataset,list_seeds, num_classes)

