import numpy as np
from typing import Generator, Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

class DataPipeline:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file using memory mapping"""
        try:
            return pd.read_csv(filepath, memory_map=True)
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
            
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data with validation and cleaning"""
        # Remove missing values
        data = data.dropna()
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Scale features
        return self.scaler.fit_transform(data)
        
    def create_batches(self, data: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Create batches using generator"""
        n_samples = len(data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield data[batch_indices]
            
    def partition_data(self, data: np.ndarray, n_partitions: int) -> List[np.ndarray]:
        """Partition data for distributed processing"""
        return np.array_split(data, n_partitions)
