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
    
    # Top configuration section
    st.header("Training Configuration")
    
    # Model configuration
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Settings")
        model_type = st.selectbox(
            "Select Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression"],
            help="""
            - Linear Regression: Basic linear model
            - Ridge Regression: Reduces overfitting with L2 regularization
            - Lasso Regression: Reduces overfitting with L1 regularization
            """
        )
        
        if model_type in ["Ridge Regression", "Lasso Regression"]:
            regularization_strength = st.slider(
                "Regularization Strength (α)",
                0.0001, 10.0, 1.0, 
                help="Controls the strength of regularization. Higher values = stronger regularization"
            )
    
    # Training configuration
    with col2:
        st.subheader("Training Settings")
        n_epochs = st.slider(
            "Number of epochs", 
            1, 100, 10,
            help="Number of complete passes through the training data"
        )
        learning_rate = st.slider(
            "Learning rate", 
            0.001, 0.5, 0.05, 
            format="%.4f",
            help="Step size for gradient updates. Lower values = more stable but slower training"
        )
        batch_size = st.slider(
            "Batch Size",
            8, 128, 32,
            help="Number of samples processed together. Larger batches = faster but may need more memory"
        )
    
    data_source = st.radio(
        "Data Source",
        ["Sample Data", "Upload Custom Data"],
        horizontal=True
    )
    
    if data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        data_path = uploaded_file if uploaded_file else "sample_data.csv"
    else:
        data_path = "sample_data.csv"
    
    # Main training interface
    if st.button("Start Training"):
        try:
            # Load and preprocess data
            with Timer("Data Loading"):
                data = data_pipeline.load_data(data_path)
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
                    gradients, avg_loss = coordinator.aggregate_gradients()
                    model.update_parameters(gradients, learning_rate)
                    
                    # Update metrics collector with loss
                    metrics_collector.update_model_metrics({'mse': avg_loss})
                    
                    # Get latest metrics
                    metrics = metrics_collector.get_latest_metrics()
                    speed_fig, resource_fig = create_metrics_charts(metrics)
                    
                    # Update UI with explanations and unique keys
                    progress_bar.progress((epoch + 1) / n_epochs)
                    
                    with metrics_container:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Epoch", f"{epoch + 1}/{n_epochs}")
                        with col2:
                            st.metric("Loss (MSE)", f"{metrics['model_metrics'].get('mse', 0):.4f}")
                        with col3:
                            st.metric("Training Speed", f"{metrics['training_speed']:.1f} samples/s")
                            
                    speed_chart.plotly_chart(speed_fig, key=f"speed_chart_{epoch}")
                    resource_chart.plotly_chart(resource_fig, key=f"resource_chart_{epoch}")

                # Show explanation only once at the start
                if epoch == 0:
                    st.info("""
                    **Metrics Guide:**
                    - Loss (MSE): Measures prediction error, lower is better
                    - Training Speed: Data processing rate
                    - Resource Usage: System utilization per worker
                    """)
                    
            # Shutdown workers
            coordinator.shutdown()
            st.success("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            st.error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()
