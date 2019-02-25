
import heapq
import abstract_optimizer
from optimization_variable import optimization_variable,discrete_var,continuous_var


class random_search(abstract_optimizer.optimizer):

    def __init__(self, optimization_variables, evaluator):
        self.optimization_variables = optimization_variables
        self.evaluator = evaluator
        self.best_seen_heap = []
    
    def step(self,*args):
        current_x = [ x.update() for x in self.optimization_variables]
        current_score = self.evaluator(current_x,*args)
        heapq.heappush(self.best_seen_heap,(current_score,current_x))