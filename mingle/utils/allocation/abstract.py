from abc import abstractmethod, ABC

class AbstractAllocation(ABC):

    @abstractmethod
    def get_weights(self, cov, omega):
        pass
