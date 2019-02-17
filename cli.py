#!/usr/bin/env python3
import argparse
from dataGenerator import DataSetCIFAR10
import model as model_factory
import analysis as analysis
import train_parameters as train_parameters_factory
import torch
import train as model_training
import torchvision.models as models
import logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Play around with a convolutional neural network on CIFAR 10')
    parser.add_argument('-modelPath', type=str, nargs='?', help="Load a model")
    parser.add_argument('-trainNew',action='store_true', default = False, help="Train thhe model")
    parser.add_argument('-trainParameters', type=str, nargs='?', help = "Training parameters json")               
    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info("Cli app started execution ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using a {} device".format(device))
    
    # Program flow parameters
    train = args.trainNew
    model_path= args.modelPath

    # ML parameters
    data = DataSetCIFAR10()
    net = model_factory.Net()
    if (train or model_path is None):
        logging.info("Starting the model training ...")
        train_parameters = train_parameters_factory.load_parameters(args.trainParameters)
        model_training.train(data,net,train_parameters)
    else:
        net.load_state_dict(torch.load(model_path))
    logging.info("Analyzing the performance of the model ...")
    analysis.model_analysis(net,data)
