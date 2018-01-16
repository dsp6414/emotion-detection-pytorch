from __future__ import print_function, division

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import scipy.misc
import pickle


def convert_fer2013_to_jpg(filepath,
                            out_file_dir,
                            num_rows=1000,
                            train_test_split=0.8):
    """

    :param filepath: path to the input fer2013 csv file
    :param num_rows: number of the rows from the fer2013 csv file to use
    :param train_test_split: the fraction of data in the training set
    :param out_file_dir: output file directory
    """
    kaggle_faces = pd.read_csv(filepath, nrows=num_rows)
    kaggle_faces = kaggle_faces.sample(frac=1).reset_index().copy()
    if not os.path.isdir(out_file_dir):
        os.mkdir(out_file_dir)
        if not os.path.isdir(os.path.join(out_file_dir, 'train')):
            os.mkdir(os.path.join(out_file_dir, 'train'))
        if not os.path.isdir(os.path.join(out_file_dir, 'test')):
            os.mkdir(os.path.join(out_file_dir, 'test'))
        for emotions in np.unique(kaggle_faces.emotion):
            if not os.path.isdir(os.path.join(out_file_dir, 'train', str(emotions))):
                os.mkdir(os.path.join(out_file_dir, 'train', str(emotions)))
            if not os.path.isdir(os.path.join(out_file_dir, 'test', str(emotions))):
                os.mkdir(os.path.join(out_file_dir, 'test', str(emotions)))
    else:
        print("Warning: the output image file already exists")

    for img_idx in range(int(train_test_split*num_rows)):
        emotion_num = kaggle_faces.emotion[img_idx]
        lin_img_array = np.array(kaggle_faces.pixels[img_idx].split(' ')).astype(int)
        new_img = np.reshape(lin_img_array, (48, 48))
        scipy.misc.imsave(os.path.join(out_file_dir,'train',str(emotion_num),'img_'+str(img_idx)+'.jpg'), new_img)

    for img_idx in range(int(train_test_split*num_rows), num_rows):
        emotion_num = kaggle_faces.emotion[img_idx]
        lin_img_array = np.array(kaggle_faces.pixels[img_idx].split(' ')).astype(int)
        new_img = np.reshape(lin_img_array, (48, 48))
        scipy.misc.imsave(os.path.join(out_file_dir,'test',str(emotion_num),'img_'+str(img_idx)+'.jpg'), new_img)


def def_default_data_transforms():
    """
    Define the default data transforms as suggested for resnet. Here, they are the same for train and test
    :return: default transformations
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def normalize_and_load_data(data_dir,
                            data_transforms=def_default_data_transforms()):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=14, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    use_gpu = torch.cuda.is_available()
    return image_datasets, dataloaders, dataset_sizes, class_names, use_gpu

def imshow_for_pytorch(inp, title=None):
    """Imshow for Tensor.
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    print(len(classes))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow_for_pytorch(out, title=[class_names[x] for x in classes])
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(0.001)


def train_model(model, criterion, optimizer, scheduler, dataloaders, use_gpu, dataset_sizes, num_epochs=15):
    since = time.time()

    # state dict contains parameters and buffers
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # epoch = one forward pass and one backward pass of all the training examples
    preds_array = []
    labels_array = []
    phase_array = []
    epoch_array = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                # loss function to penalize some criterion
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                preds_array.append(preds)
                labels_array.append(labels)
                phase_array.append(phase)
                epoch_array.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_array, phase_array, labels_array, preds_array


def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow_for_pytorch(inputs.cpu().data[j])

            if images_so_far == num_images:
                return



def visualize_model_20img(model, num_images=20):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow_for_pytorch(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def run_model(dataloaders, use_gpu, dataset_sizes, pickle_file_name):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft, epoch_array, phase_array, labels_array, preds_array = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,dataloaders,use_gpu,dataset_sizes,
                           num_epochs=15)

    with open(pickle_file_name, 'w') as f:
        pickle.dump([model_ft, epoch_array, phase_array, labels_array, preds_array], f)

    return model_ft, epoch_array, phase_array, labels_array, preds_array

def run_model_multiclass(dataloaders, use_gpu, dataset_sizes, pickle_file_name, num_classes=7):
    model = models.resnet18(pretrained=True)
    inputs, labels = next(iter(dataloaders['train']))
    if use_gpu:
        model = model.cuda()
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = model(inputs)
    print(outputs.size())

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    if use_gpu:
        model = model.cuda()

    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    model,epoch_array, phase_array, labels_array, preds_array = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, use_gpu, dataset_sizes,
                           num_epochs=10)

    with open(pickle_file_name, 'w') as f:
        pickle.dump([model, epoch_array, phase_array, labels_array, preds_array], f)

    return model, epoch_array, phase_array, labels_array, preds_array


if __name__=="__main__":
    image_datasets, dataloaders, dataset_sizes, class_names, use_gpu = normalize_and_load_data()
    run_model(dataloaders, use_gpu, dataset_sizes)
