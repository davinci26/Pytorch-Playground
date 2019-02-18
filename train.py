import torch
import torchvision
import datetime
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import logging


def train( dataset, net, training_parameters, save=True):

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=training_parameters["learning_rate"], momentum=training_parameters["momentum"])

    for epoch in range(training_parameters["epoch"]):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataset.train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    logging.info('Finished Training')
    
    if save:
        logging.info('Saving the model')
        # Generate unique string for the file
        unix_time = '%.0f' % time.mktime(datetime.date.today().timetuple())
        # save model
        torch.save(net.state_dict(),'models/{}-{}.model'.format(net.__class__.__name__,unix_time))
        # save parameters next to the model
        params_filename = 'models/{}-{}-parameters.json'.format(net.__class__.__name__,unix_time)
        with open(params_filename, 'w') as fp:
            json.dump(training_parameters, fp)