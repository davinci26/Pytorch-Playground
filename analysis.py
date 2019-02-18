import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import logging

def log_prediction(true_label, predicted_label, predicted_label_index, output_layer):
    output_layer_pretty_string = ' '.join('%.3f' % out for out in output_layer)
    logging.info('Ground truth: {} Prediction: {} Argmax: {} Output layer: {}'.format(true_label,predicted_label,predicted_label_index,output_layer_pretty_string))


def make_predictions(batch_number, batch_size, dataset_iterator, classes, net):
    with torch.no_grad():
        batch_counter = 0
        for data in dataset_iterator:
            if batch_counter == batch_number:
                return
            batch_counter += 1
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            logging.info("Making prediction for branch {}/{} with branch size {}".format(batch_counter,batch_number,batch_size))
            for j in range(batch_size):
                ground_truth =  '%5s' % classes[labels[j]]
                prediction_label = '%5s' % classes[predicted[j]]
                output_layer = outputs[j].tolist()
                log_prediction(ground_truth,prediction_label,predicted[j],output_layer)


def calculate_loss(dataset_iterator,net,criterion):
    running_loss = 0
    item_counter = 0
    with torch.no_grad():
        for data in dataset_iterator:
            # get the inputs
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            item_counter += 1
    return running_loss / item_counter

def total_accuracy(dataset_iterator,net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset_iterator:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuraccy = 100 * correct / total
    logging.info('Accuracy of the network on the %d test images: %d %%' % (total, accuraccy))
    return accuraccy

def per_class_accuracy(batch_size,dataset_iterator, classes, net):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataset_iterator:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    accuracy_per_class  = []
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        accuracy_per_class.append(accuracy)
        logging.info('Accuracy of %5s : %2d %%' % (classes[i], accuracy))
    return accuracy_per_class

def model_analysis(net, data, batch_size = 4):
    test_loader = data.valid_loader
    train_loader = data.train_loader
    classes = data.classes
    make_predictions(1,batch_size,test_loader,classes,net)
    logging.info("Calculating the accuraccy in the whole dataset...")
    total_accuracy(test_loader,net)
    logging.info("Calculating the accuraccy per class...")
    per_class_accuracy(batch_size,test_loader,classes,net)
    logging.info("Cross Entropy loss on training set {:.3f}".format(calculate_loss(train_loader,net,nn.CrossEntropyLoss())))
    logging.info("Cross Entropy loss on validation set {:.3f}".format(calculate_loss(test_loader,net,nn.CrossEntropyLoss())))