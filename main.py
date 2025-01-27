import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
import logging

from coordinator import TrainingCoordinator
from data_pipeline import DataPipeline
from models import LinearRegression
from utils import setup_logging, Timer
from metrics import MetricsCollector

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def initialize_system():
    coordinator = TrainingCoordinator(n_workers=4)
    data_pipeline = DataPipeline(batch_size=32)
    model = LinearRegression()
    metrics_collector = MetricsCollector()
    return coordinator, data_pipeline, model, metrics_collector

def create_metrics_charts(metrics: Dict[str, Any]):
    """Create charts for displaying metrics"""
    # Training speed chart
    speed_fig = go.Figure(data=go.Scatter(
        y=[metrics['training_speed']],
        mode='lines+markers',
        name='Training Speed'
    ))
    speed_fig.update_layout(title='Training Speed (samples/second)')
    
    # Resource utilization chart
    resource_fig = go.Figure(data=[
        go.Bar(name='Memory Usage (MB)', 
               x=['Memory'], 
               y=[metrics['avg_memory_usage']]),
        go.Bar(name='Compute Time (s)', 
               x=['Compute'], 
               y=[metrics['avg_compute_time']])
    ])
    resource_fig.update_layout(title='Resource Utilization')
    
    return speed_fig, resource_fig

def main():
    st.title("Distributed Machine Learning System")
    
    # Initialize system components
    coordinator, data_pipeline, model, metrics_collector = initialize_system()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    n_epochs = st.sidebar.slider("Number of epochs", 1, 100, 10)
    learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01)
    
    # Main training interface
    if st.button("Start Training"):
        try:
            # Load and preprocess data
            with Timer("Data Loading"):
                data = data_pipeline.load_data("data.csv")
                processed_data = data_pipeline.preprocess_data(data)
                
            # Initialize model
            model.initialize_parameters(processed_data.shape[1] - 1)
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Create metrics placeholders
            metrics_container = st.container()
            speed_chart = st.empty()
            resource_chart = st.empty()
            
            # Training loop
            for epoch in range(n_epochs):
                with Timer(f"Epoch {epoch}"):
                    # Partition data
                    partitions = data_pipeline.partition_data(processed_data, coordinator.n_workers)
                    
                    # Initialize workers
                    coordinator.initialize_workers(model, partitions)
                    
                    # Training step
                    coordinator.broadcast_parameters(model.parameters)
                    gradients = coordinator.aggregate_gradients()
                    model.update_parameters(gradients, learning_rate)
                    
                    # Update metrics
                    metrics = metrics_collector.get_latest_metrics()
                    speed_fig, resource_fig = create_metrics_charts(metrics)
                    
                    # Update UI
                    progress_bar.progress((epoch + 1) / n_epochs)
                    with metrics_container:
                        st.write(f"Epoch {epoch + 1}/{n_epochs}")
                        st.write(f"Loss: {metrics['model_metrics'].get('mse', 0):.4f}")
                    
                    speed_chart.plotly_chart(speed_fig)
                    resource_chart.plotly_chart(resource_fig)
                    
            # Shutdown workers
            coordinator.shutdown()
            st.success("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            st.error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()
