import numpy as np
from typing import Dict, Any
import logging
from models import BaseModel

class Worker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.logger = logging.getLogger(__name__)
        
    def process_batch(self, model: BaseModel, batch_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a batch of data and compute gradients"""
        try:
            gradients = model.compute_gradients(batch_data)
            self.logger.debug(f"Worker {self.worker_id} computed gradients successfully")
            return gradients
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} failed to compute gradients: {str(e)}")
            raise
            
    def update_parameters(self, parameters: Dict[str, np.ndarray]):
        """Update local model parameters"""
        self.current_parameters = parameters
