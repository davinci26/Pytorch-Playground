#!/usr/bin/env python3
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname( __file__ ), ".."))
from core.optimizers.optimization_variable import discrete_var,continuous_var
from core.optimizers.random_search import random_search

def complex_evaluator(x):
    return 1

class optimizer_tests(unittest.TestCase):

    def test_optimizer_constructor_with_continous_and_discrete_variables_should_not_throw(self):

        # Arrange
        def plain_evaluator(variable_list):
            score =  sum(variable_list)
            return score

        var1 = continuous_var(0,-1,1)
        var2 = discrete_var(0,[0,1,2,3,4,5,6])
        # Act & Assert
        self.assertIsNotNone(random_search([var1,var2],plain_evaluator))
    
    def test_random_search_steps_with_simple_evaluator(self):     
        # Arrange
        def plain_evaluator(variable_list):
            score =  sum(variable_list) 
            return score

        var1 = continuous_var(0,-1,1)
        var2 = discrete_var(0,[0,1,2,3,4,5,6])
        optimizer = random_search([var1,var2],plain_evaluator)

         # Act
        for _ in range(0,1):
            optimizer.step()

        # Assert 
        self.assertEqual(optimizer.best_seen_heap[0][0], var1.initial_value + var2.initial_value)
    

    def test_random_search_steps_with_complex_evaluator(self):     
        # Arrange
        def complex_evaluator(variable_list,other_parameter):
            self.assertEqual('a',other_parameter)
            score =  sum(variable_list) 
            return score

        var1 = continuous_var(0,-1,1)
        var2 = discrete_var(0,[0,1,2,3,4,5,6])
        optimizer = random_search([var1,var2],complex_evaluator)
        
         # Act
        for _ in range(0,1):
            optimizer.step('a')

        # Assert 
        self.assertEqual(optimizer.best_seen_heap[0][0], var1.initial_value + var2.initial_value)
    

if __name__ == '__main__':
    unittest.main()