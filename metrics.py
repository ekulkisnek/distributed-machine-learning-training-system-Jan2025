from typing import Dict, Any
import time
import numpy as np
import logging

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'training_speed': [],
            'memory_usage': {},
            'compute_time': {},
            'model_metrics': []
        }
        self.logger = logging.getLogger(__name__)
        
    def update_worker_metrics(self, worker_id: int, compute_time: float, 
                            memory_usage: float):
        """Update metrics for a worker"""
        self.metrics['compute_time'][worker_id] = compute_time
        self.metrics['memory_usage'][worker_id] = memory_usage
        
    def update_training_speed(self, samples_per_second: float):
        """Update training speed metric"""
        self.metrics['training_speed'].append(samples_per_second)
        
    def update_model_metrics(self, metrics: Dict[str, float]):
        """Update model performance metrics"""
        self.metrics['model_metrics'].append(metrics)
        
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics for visualization"""
        return {
            'training_speed': self.metrics['training_speed'][-1] if self.metrics['training_speed'] else 0,
            'avg_memory_usage': np.mean(list(self.metrics['memory_usage'].values())),
            'avg_compute_time': np.mean(list(self.metrics['compute_time'].values())),
            'model_metrics': self.metrics['model_metrics'][-1] if self.metrics['model_metrics'] else {}
        }
