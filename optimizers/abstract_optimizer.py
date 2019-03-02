from abc import ABC, abstractmethod

class optimizer(ABC):
    """
        def evaluator(x):
            print(x)
            return 1

        var1 = continuous_var(0,-1,1)
        var2 = discrete_var(0,[0,1,2,3,4,5,6])
        search = optimizer([var1,var2],evaluator)

        for i in range(100):
            search.step()
    """
    @abstractmethod
    def __init__(self, variables):
        self.variables = variables
    
    @abstractmethod
    def step(self):
        """ Single optimization step
        """
