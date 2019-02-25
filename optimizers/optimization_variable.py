import random
from abc import ABC, abstractmethod
 
class optimization_variable(ABC):
    @abstractmethod
    def update(self):
        """ Update the optimization variable
        """

class discrete_var(optimization_variable):

    def __init__(self, initial_value, possible_values):
        self.initial_value = initial_value
        self.possible_values = possible_values
        self.current_value = self.initial_value
        self.value_length = len(self.possible_values)
    
    def update(self):
        current_index = random.randint(0,self.value_length - 1)
        self.current_value = self.possible_values[current_index]
        return self.current_value


class continuous_var(optimization_variable):

    def __init__(self, initial_value, lower_bound, upper_bound):
        self.initial_value = initial_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.current_value = self.initial_value
    
    def update(self):
        self.current_value = random.uniform(self.lower_bound,self.upper_bound)
        return self.current_value