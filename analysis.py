import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from visualize import imshow
import logging

def log_prediction(true_label, predicted_label, predicted_label_index, output_layer):
    output_layer_pretty_string = ' '.join('%.3f' % out for out in output_layer)
    logging.info('Ground truth: {} Prediction: {} Argmax: {} Output layer: {}'.format(true_label,predicted_label,predicted_label_index,output_layer_pretty_string))


def model_analysis(net, data, prediction_number = 4):
    _, _, _, testloader, classes = data.get_all()

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    logging.info("Making predictions for {} number of images".format(prediction_number))
    for j in range(prediction_number):
        ground_truth =  '%5s' % classes[labels[j]]
        prediction_label = '%5s' % classes[predicted[j]]
        output_layer = outputs[j].tolist()
        log_prediction(ground_truth,prediction_label,predicted[j],output_layer)
    
    logging.info("Calculating the accuraccy in the whole dataset")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    logging.info("Calculating the accuraccy per class")
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
        logging.info('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
    del dataiter
    # %%%%%%INVISIBLE_CODE_BLOCK%%%%%%