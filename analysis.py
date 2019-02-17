import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from visualize import imshow
import logging


def model_analysis(net, data):
    _, _, _, testloader, classes = data.get_all()

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    ground_truth_string = ' '.join('%5s' % classes[labels[j]] for j in range(4))
    logging.info('Ground truth: {}'.format(ground_truth_string))

    ########################################################################
    # Okay, now let us see what the neural network thinks these examples above are:

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    prediction_string = ' '.join('%5s' % classes[predicted[j]] for j in range(4))
    logging.info('Predicted: {}'.format(prediction_string))

    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    ########################################################################
    # That looks waaay better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        logging.info('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(device)
    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
    del dataiter
    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%