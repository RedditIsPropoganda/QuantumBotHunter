import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from ai_core.federated_learning.federated_trainer import start_federated_learning
from pipelines.data_pipeline import DataPipeline
from models.bot_detection_model import BotDetectionModel
from services.api.bot_detection_api import app as api_app
import uvicorn
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config: Dict[str, Any]) -> None:
    """Train the bot detection model using the provided configuration."""
    try:
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test = pipeline.run_pipeline(config['data_path'], config['target_column'])
        
        model = BotDetectionModel(input_dim=X_train.shape[1], hidden_dim=config['hidden_dim'], output_dim=1)
        model.fit(X_train, y_train, epochs=config['epochs'])
        
        loss, accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Model trained. Test accuracy: {accuracy:.2f}")
        
        # Save the model
        torch.save(model.state_dict(), config['model_save_path'])
        logging.info(f"Model saved to {config['model_save_path']}")
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")

def run_federated_learning(config: Dict[str, Any]) -> None:
    """Start the federated learning process using the provided configuration."""
    try:
        start_federated_learning(config)
    except Exception as e:
        logging.error(f"Error during federated learning: {str(e)}")

def start_api(config: Dict[str, Any]) -> None:
    """Start the API service using the provided configuration."""
    try:
        uvicorn.run(api_app, host=config['api_host'], port=config['api_port'])
    except Exception as e:
        logging.error(f"Error starting API: {str(e)}")

def launch_dashboard(config: Dict[str, Any]) -> None:
    """Launch the dashboard using the provided configuration."""
    try:
        # This is a placeholder. Implement dashboard launch logic here.
        logging.info(f"Launching dashboard at http://{config['dashboard_host']}:{config['dashboard_port']}")
    except Exception as e:
        logging.error(f"Error launching dashboard: {str(e)}")

def main() -> None:
    """Main function to parse arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(description="QuantumBotHunter CLI")
    parser.add_argument('action', choices=['train', 'federated', 'api', 'dashboard'], 
                        help='Action to perform: train model, start federated learning, run API, or launch dashboard')
    parser.add_argument('--config', default='config/main_config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.action == 'train':
        train_model(config)
    elif args.action == 'federated':
        run_federated_learning(config)
    elif args.action == 'api':
        start_api(config)
    elif args.action == 'dashboard':
        launch_dashboard(config)

if __name__ == "__main__":
    main()
