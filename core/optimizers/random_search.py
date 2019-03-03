
import heapq
import core.optimizers.abstract_optimizer
from core.optimizers.optimization_variable import optimization_variable,discrete_var,continuous_var


class random_search(core.optimizers.abstract_optimizer.optimizer):

    def __init__(self, optimization_variables, evaluator):
        super().__init__(optimization_variables)
        self.evaluator = evaluator
        self.best_seen_heap = []
    
    def step(self,*args):
        current_x = [ x.current_value for x in self.variables]
        current_score = self.evaluator(current_x,*args)
        heapq.heappush(self.best_seen_heap,(current_score,current_x))
        for x in self.variables:
            x.update()