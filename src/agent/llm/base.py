from abc import ABC, abstractmethod
from typing import Any
class BaseProvider(ABC):

    @abstractmethod
    def create_client(self , **kwargs:Any) -> Any:
        """
        Abstract method to create a client for the provider. 
        This method should be implemented by all subclasses to return an instance of the provider's client.
        """
        pass

