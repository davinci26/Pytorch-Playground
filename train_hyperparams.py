import numpy as np
import logging
import copy
import analysis as analysis
from optimizers.random_search import random_search
from optimizers.optimization_variable import optimization_variable,discrete_var,continuous_var
import train as model_training

def hyperparameter_optimization(dataset, net, training_parameters, save=True):
    var_learning_rate= discrete_var(0,np.logspace(0.0001,0.1))
    var_momentum = continuous_var(0.85,0.8,1)
    search = random_search([var_learning_rate,var_momentum],model_training.train)
    current_net = copy.deepcopy(net)
    for _ in range(training_parameters["hyperparameter_optimization"]["steps"]):
        search.step(dataset, current_net, training_parameters,False)