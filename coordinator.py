import multiprocessing as mp
from typing import List, Dict, Any
import numpy as np
import time
import logging
from models import BaseModel
from metrics import MetricsCollector

class TrainingCoordinator:
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.workers = []
        self.parameter_queue = mp.Queue()
        self.gradient_queue = mp.Queue()
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)

    def initialize_workers(self, model: BaseModel, data_partitions: List[np.ndarray]):
        """Initialize worker processes with data partitions"""
        for i in range(self.n_workers):
            worker = mp.Process(
                target=self._worker_process,
                args=(i, model, data_partitions[i], 
                      self.parameter_queue, self.gradient_queue)
            )
            self.workers.append(worker)
            worker.start()

    def _worker_process(self, worker_id: int, model: BaseModel, 
                       data: np.ndarray, param_queue: mp.Queue, 
                       grad_queue: mp.Queue):
        """Worker process function"""
        while True:
            # Get latest parameters
            parameters = param_queue.get()
            if parameters is None:  # Termination signal
                break

            # Compute gradients and loss
            predictions = np.dot(data[:, :-1], parameters['weights']) + parameters['bias']
            loss = np.mean((predictions - data[:, -1]) ** 2)
            gradients = model.compute_gradients(data)
            grad_queue.put((worker_id, gradients, loss))

            # Collect metrics
            self.metrics_collector.update_worker_metrics(
                worker_id=worker_id,
                compute_time=time.time(),
                memory_usage=self._get_process_memory()
            )

    def _get_process_memory(self) -> float:
        """Get current process memory usage"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def aggregate_gradients(self) -> Dict[str, np.ndarray]:
        """Aggregate gradients from all workers"""
        gradients = []
        losses = []
        for _ in range(self.n_workers):
            worker_id, grad, loss = self.gradient_queue.get()
            gradients.append(grad)
            losses.append(loss)
        average_loss = np.mean(losses)
        return {k: np.mean([g[k] for g in gradients], axis=0) 
                for k in gradients[0].keys()}, average_loss

    def broadcast_parameters(self, parameters: Dict[str, np.ndarray]):
        """Broadcast parameters to all workers"""
        for _ in range(self.n_workers):
            self.parameter_queue.put(parameters)

    def shutdown(self):
        """Shutdown all workers"""
        for _ in range(self.n_workers):
            self.parameter_queue.put(None)
        for worker in self.workers:
            worker.join()