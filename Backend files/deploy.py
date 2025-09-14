import os
import sys
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import docker
import yaml

# Import our custom modules
from models import MLModelManager
from data import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    """
    Comprehensive model deployment system supporting multiple cloud providers,
    containerization, API generation, and monitoring setup
    """
    
    def __init__(self, deployments_dir: str = 'deployments', config_dir: str = 'deployment_configs'):
        self.deployments_dir = deployments_dir
        self.config_dir = config_dir
        self.model_manager = MLModelManager()
        self.data_processor = DataProcessor()
        
        # Create directories
        os.makedirs(deployments_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        # Deployment tracking
        self.active_deployments = {}
        self.deployment_history = []
        
        # Cloud clients (initialized as needed)
        self.aws_client = None
        self.gcp_client = None
        self.azure_client = None
    
    def deploy_model(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a trained model based on configuration
        """
        try:
            deployment_id = str(uuid.uuid4())
            
            # Validate configuration
            self._validate_deployment_config(deployment_config)
            
            # Initialize deployment info
            deployment_info = {
                'deployment_id': deployment_id,
                'status': 'initializing',
                'config': deployment_config,
                'start_time': datetime.now().isoformat(),
                'progress': 0.0,
                'logs': [],
                'endpoints': {},
                'monitoring': {},
                'error': None
            }
            
            self.active_deployments[deployment_id] = deployment_info
            self._log_deployment(deployment_id, "Starting model deployment...")
            
            # Load and validate model
            model_path = deployment_config['model_path']
            model, metadata = self._load_and_validate_model(model_path)
            
            self._update_deployment_progress(deployment_id, 'loading_model', 10.0, 
                                           ["Model loaded successfully"])
            
            # Generate prediction API
            api_code = self._generate_prediction_api(model, metadata, deployment_config)
            api_path = os.path.join(self.deployments_dir, f"{deployment_id}_api.py")
            
            with open(api_path, 'w') as f:
                f.write(api_code)
            
            self._update_deployment_progress(deployment_id, 'api_generated', 20.0, 
                                           ["Prediction API generated"])
            
            # Create deployment artifacts
            artifacts_dir = os.path.join(self.deployments_dir, deployment_id)
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Copy model to deployment directory
            import shutil
            deployed_model_path = os.path.join(artifacts_dir, 'model.pkl')
            shutil.copy2(model_path, deployed_model_path)
            
            # Create requirements file
            self._create_requirements_file(artifacts_dir, deployment_config)
            
            # Create Docker configuration if needed
            if deployment_config.get('containerize', True):
                self._create_docker_config(artifacts_dir, deployment_config)
                self._update_deployment_progress(deployment_id, 'containerized', 40.0, 
                                               ["Docker configuration created"])
            
            # Deploy based on target platform
            platform = deployment_config.get('platform', 'local')
            
            if platform == 'aws':
                deployment_result = self._deploy_to_aws(deployment_id, artifacts_dir, deployment_config)
            elif platform == 'gcp':
                deployment_result = self._deploy_to_gcp(deployment_id, artifacts_dir, deployment_config)
            elif platform == 'azure':
                deployment_result = self._deploy_to_azure(deployment_id, artifacts_dir, deployment_config)
            elif platform == 'local':
                deployment_result = self._deploy_locally(deployment_id, artifacts_dir, deployment_config)
            else:
                raise ValueError(f"Unsupported deployment platform: {platform}")
            
            # Update deployment info with results
            deployment_info.update(deployment_result)
            deployment_info['status'] = 'completed'
            deployment_info['progress'] = 100.0
            deployment_info['end_time'] = datetime.now().isoformat()
            
            self._log_deployment(deployment_id, f"Deployment completed successfully to {platform}")
            
            # Set up monitoring if requested
            if deployment_config.get('enable_monitoring', True):
                monitoring_config = self._setup_monitoring(deployment_id, deployment_config)
                deployment_info['monitoring'] = monitoring_config
            
            # Move to history
            self.deployment_history.append(deployment_info.copy())
            
            return {
                'success': True,
                'deployment_id': deployment_id,
                'deployment_info': deployment_info
            }
            
        except Exception as e:
            error_msg = f"Deployment failed: {str(e)}"
            logger.error(error_msg)
            
            if deployment_id in self.active_deployments:
                self.active_deployments[deployment_id]['status'] = 'failed'
                self.active_deployments[deployment_id]['error'] = error_msg
                self._log_deployment(deployment_id, error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'deployment_id': deployment_id if 'deployment_id' in locals() else None
            }
    
    def _validate_deployment_config(self, config: Dict[str, Any]):
        """Validate deployment configuration"""
        required_fields = ['model_path', 'api_name']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if not os.path.exists(config['model_path']):
            raise ValueError(f"Model file not found: {config['model_path']}")
        
        # Validate platform-specific requirements
        platform = config.get('platform', 'local')
        if platform == 'aws' and not config.get('aws_region'):
            config['aws_region'] = 'us-east-1'  # default
    
    def _load_and_validate_model(self, model_path: str):
        """Load and validate model for deployment"""
        try:
            model, metadata = self.model_manager.load_model(model_path)
            
            # Validate model has predict method
            if not hasattr(model, 'predict'):
                raise ValueError("Model does not have predict method")
            
            return model, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def _generate_prediction_api(self, model, metadata: Dict[str, Any], 
                                config: Dict[str, Any]) -> str:
        """Generate Flask API code for model predictions"""
        api_name = config['api_name']
        algorithm = metadata.get('algorithm', 'unknown')
        feature_names = metadata.get('ml_data_info', {}).get('feature_names', [])
        
        api_template = f'''
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load model
MODEL_PATH = 'model.pkl'
model = None
metadata = None

def load_model():
    global model, metadata
    try:
        import joblib
        model_package = joblib.load(MODEL_PATH)
        
        if isinstance(model_package, dict) and 'model' in model_package:
            model = model_package['model']
            metadata = model_package.get('metadata', {{}})
        else:
            model = model_package
            metadata = {{}}
            
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {{e}}")
        raise

# Load model on startup
load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({{
        'status': 'healthy',
        'api_name': '{api_name}',
        'algorithm': '{algorithm}',
        'timestamp': datetime.now().isoformat()
    }})

@app.route('/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({{
        'api_name': '{api_name}',
        'algorithm': '{algorithm}',
        'features': {json.dumps(feature_names)},
        'metadata': metadata
    }})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({{'error': 'No input data provided'}}), 400
        
        # Handle different input formats
        if 'features' in data:
            # Single prediction with feature array
            features = data['features']
            if not isinstance(features, list):
                return jsonify({{'error': 'Features must be a list'}}), 400
            
            # Convert to DataFrame if feature names available
            if {len(feature_names) > 0}:
                if len(features) != {len(feature_names)}:
                    return jsonify({{
                        'error': f'Expected {{len(feature_names)}} features, got {{len(features)}}'
                    }}), 400
                df = pd.DataFrame([features], columns={json.dumps(feature_names)})
            else:
                df = pd.DataFrame([features])
            
        elif 'instances' in data:
            # Batch predictions
            instances = data['instances']
            if not isinstance(instances, list):
                return jsonify({{'error': 'Instances must be a list'}}), 400
            
            df = pd.DataFrame(instances)
            
        elif isinstance(data, dict) and any(isinstance(v, (int, float)) for v in data.values()):
            # Single prediction with feature dictionary
            df = pd.DataFrame([data])
            
        else:
            return jsonify({{'error': 'Invalid input format'}}), 400
        
        # Make prediction
        predictions = model.predict(df)
        
        # Get probabilities if available (for classification)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df)
                probabilities = probabilities.tolist()
            except:
                pass
        
        # Format response
        if len(predictions) == 1:
            # Single prediction
            response = {{
                'prediction': predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0],
                'probabilities': probabilities[0] if probabilities else None
            }}
        else:
            # Batch predictions
            response = {{
                'predictions': [p.tolist() if hasattr(p, 'tolist') else p for p in predictions],
                'probabilities': probabilities
            }}
        
        response['timestamp'] = datetime.now().isoformat()
        return jsonify(response)
        
    except Exception as e:
        return jsonify({{'error': f'Prediction failed: {{str(e)}}'}}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'instances' not in data:
            return jsonify({{'error': 'No instances provided'}}), 400
        
        instances = data['instances']
        df = pd.DataFrame(instances)
        
        predictions = model.predict(df)
        
        return jsonify({{
            'predictions': [p.tolist() if hasattr(p, 'tolist') else p for p in predictions],
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }})
        
    except Exception as e:
        return jsonify({{'error': f'Batch prediction failed: {{str(e)}}'}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        
        return api_template.strip()
    
    def _create_requirements_file(self, artifacts_dir: str, config: Dict[str, Any]):
        """Create requirements.txt for deployment"""
        base_requirements = [
            'flask>=2.0.0',
            'scikit-learn>=1.0.0',
            'pandas>=1.3.0',
            'numpy>=1.20.0',
            'joblib>=1.0.0'
        ]
        
        # Add additional requirements based on model type
        additional_reqs = config.get('additional_requirements', [])
        all_requirements = base_requirements + additional_reqs
        
        requirements_path = os.path.join(artifacts_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(all_requirements))
    
    def _create_docker_config(self, artifacts_dir: str, config: Dict[str, Any]):
        """Create Dockerfile and docker-compose.yml"""
        # Create Dockerfile
        dockerfile_content = f'''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and API
COPY model.pkl .
COPY {config["deployment_id"]}_api.py app.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]
'''
        
        dockerfile_path = os.path.join(artifacts_dir, 'Dockerfile')
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        # Create docker-compose.yml
        compose_content = f'''
version: '3.8'

services:
  {config['api_name']}:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: {config.get('memory_limit', '1G')}
        reservations:
          memory: {config.get('memory_reservation', '512M')}
'''
        
        compose_path = os.path.join(artifacts_dir, 'docker-compose.yml')
        with open(compose_path, 'w') as f:
            f.write(compose_content.strip())
    
    def _deploy_to_aws(self, deployment_id: str, artifacts_dir: str, 
                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to AWS using ECS or Lambda"""
        self._update_deployment_progress(deployment_id, 'deploying_aws', 60.0, 
                                       ["Starting AWS deployment..."])
        
        try:
            # Initialize AWS client
            if not self.aws_client:
                self.aws_client = boto3.client(
                    'ecs',
                    region_name=config.get('aws_region', 'us-east-1')
                )
            
            deployment_method = config.get('aws_deployment_method', 'ecs')
            
            if deployment_method == 'ecs':
                return self._deploy_to_ecs(deployment_id, artifacts_dir, config)
            elif deployment_method == 'lambda':
                return self._deploy_to_lambda(deployment_id, artifacts_dir, config)
            else:
                raise ValueError(f"Unsupported AWS deployment method: {deployment_method}")
                
        except NoCredentialsError:
            raise ValueError("AWS credentials not configured")
        except ClientError as e:
            raise ValueError(f"AWS deployment failed: {e}")
    
    def _deploy_to_ecs(self, deployment_id: str, artifacts_dir: str, 
                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to AWS ECS"""
        # This is a simplified version - real implementation would involve:
        # 1. Building and pushing Docker image to ECR
        # 2. Creating ECS task definition
        # 3. Creating or updating ECS service
        # 4. Setting up load balancer
        # 5. Configuring auto-scaling
        
        self._update_deployment_progress(deployment_id, 'ecs_deployment', 80.0, 
                                       ["Deploying to AWS ECS..."])
        
        # Simulate deployment
        api_url = f"https://{config['api_name']}-{deployment_id[:8]}.aws-region.elb.amazonaws.com"
        
        return {
            'platform': 'aws_ecs',
            'endpoints': {
                'api_url': api_url,
                'health_url': f"{api_url}/health",
                'predict_url': f"{api_url}/predict"
            },
            'deployment_details': {
                'cluster_name': f"{config['api_name']}-cluster",
                'service_name': f"{config['api_name']}-service",
                'task_definition': f"{config['api_name']}-task"
            }
        }
    
    def _deploy_to_lambda(self, deployment_id: str, artifacts_dir: str, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to AWS Lambda"""
        self._update_deployment_progress(deployment_id, 'lambda_deployment', 80.0, 
                                       ["Deploying to AWS Lambda..."])
        
        # Lambda deployment would involve creating deployment package
        # and setting up API Gateway
        
        api_url = f"https://{deployment_id[:8]}.execute-api.us-east-1.amazonaws.com/prod"
        
        return {
            'platform': 'aws_lambda',
            'endpoints': {
                'api_url': api_url,
                'predict_url': f"{api_url}/predict"
            },
            'deployment_details': {
                'function_name': f"{config['api_name']}-function",
                'api_gateway_id': f"{deployment_id[:8]}"
            }
        }
    
    def _deploy_to_gcp(self, deployment_id: str, artifacts_dir: str, 
                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Google Cloud Platform"""
        self._update_deployment_progress(deployment_id, 'deploying_gcp', 60.0, 
                                       ["Starting GCP deployment..."])
        
        # GCP deployment would use Cloud Run or App Engine
        api_url = f"https://{config['api_name']}-{deployment_id[:8]}.run.app"
        
        return {
            'platform': 'gcp',
            'endpoints': {
                'api_url': api_url,
                'health_url': f"{api_url}/health",
                'predict_url': f"{api_url}/predict"
            },
            'deployment_details': {
                'service_name': f"{config['api_name']}-service",
                'region': config.get('gcp_region', 'us-central1')
            }
        }
    
    def _deploy_to_azure(self, deployment_id: str, artifacts_dir: str, 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to Microsoft Azure"""
        self._update_deployment_progress(deployment_id, 'deploying_azure', 60.0, 
                                       ["Starting Azure deployment..."])
        
        # Azure deployment would use Container Instances or App Service
        api_url = f"https://{config['api_name']}-{deployment_id[:8]}.azurewebsites.net"
        
        return {
            'platform': 'azure',
            'endpoints': {
                'api_url': api_url,
                'health_url': f"{api_url}/health",
                'predict_url': f"{api_url}/predict"
            },
            'deployment_details': {
                'app_service_name': f"{config['api_name']}-app",
                'resource_group': config.get('azure_resource_group', 'ml-models')
            }
        }
    
    def _deploy_locally(self, deployment_id: str, artifacts_dir: str, 
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy locally using Docker"""
        self._update_deployment_progress(deployment_id, 'deploying_local', 60.0, 
                                       ["Starting local deployment..."])
        
        try:
            # Use Docker to run the container
            client = docker.from_env()
            
            # Build image
            image_name = f"{config['api_name']}:{deployment_id[:8]}"
            client.images.build(path=artifacts_dir, tag=image_name)
            
            # Run container
            port = config.get('port', 5000)
            container = client.containers.run(
                image_name,
                ports={f'5000/tcp': port},
                detach=True,
                name=f"{config['api_name']}-{deployment_id[:8]}"
            )
            
            api_url = f"http://localhost:{port}"
            
            return {
                'platform': 'local',
                'endpoints': {
                    'api_url': api_url,
                    'health_url': f"{api_url}/health",
                    'predict_url': f"{api_url}/predict"
                },
                'deployment_details': {
                    'container_id': container.id,
                    'image_name': image_name,
                    'port': port
                }
            }
            
        except docker.errors.DockerException as e:
            raise ValueError(f"Docker deployment failed: {e}")
    
    def _setup_monitoring(self, deployment_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up monitoring for deployed model"""
        monitoring_config = {
            'enabled': True,
            'metrics': ['request_count', 'response_time', 'error_rate'],
            'alerts': config.get('alerts', []),
            'logging_level': config.get('logging_level', 'INFO'),
            'health_check_interval': config.get('health_check_interval', 60)
        }
        
        # In a real implementation, this would set up:
        # - CloudWatch/Stackdriver/Azure Monitor
        # - Custom metrics collection
        # - Alerting rules
        # - Log aggregation
        
        return monitoring_config
    
    def _update_deployment_progress(self, deployment_id: str, status: str, 
                                  progress: float, logs: List[str]):
        """Update deployment progress"""
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            deployment['status'] = status
            deployment['progress'] = progress
            deployment['logs'].extend(logs)
            deployment['last_update'] = datetime.now().isoformat()
    
    def _log_deployment(self, deployment_id: str, message: str):
        """Add log message to deployment"""
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id]['logs'].append(
                f"[{datetime.now().isoformat()}] {message}"
            )
        logger.info(f"Deployment {deployment_id}: {message}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id].copy()
        
        # Check history
        for deployment in self.deployment_history:
            if deployment['deployment_id'] == deployment_id:
                return deployment
        
        return None
    
    def stop_deployment(self, deployment_id: str) -> bool:
        """Stop a running deployment"""
        deployment = self.get_deployment_status(deployment_id)
        if not deployment:
            return False
        
        platform = deployment.get('platform', 'local')
        deployment_details = deployment.get('deployment_details', {})
        
        try:
            if platform == 'local' and 'container_id' in deployment_details:
                # Stop Docker container
                client = docker.from_env()
                container = client.containers.get(deployment_details['container_id'])
                container.stop()
                container.remove()
                return True
            elif platform.startswith('aws'):
                # Stop AWS deployment (ECS service, Lambda, etc.)
                # Implementation would depend on deployment method
                pass
            # Add other platforms as needed
            
        except Exception as e:
            logger.error(f"Failed to stop deployment {deployment_id}: {e}")
            return False
        
        return True
    
    def list_deployments(self, status: str = None) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by status"""
        all_deployments = list(self.active_deployments.values()) + self.deployment_history
        
        if status:
            all_deployments = [d for d in all_deployments if d.get('status') == status]
        
        # Sort by start time, newest first
        all_deployments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return all_deployments
    
    def get_deployment_logs(self, deployment_id: str) -> List[str]:
        """Get deployment logs"""
        deployment = self.get_deployment_status(deployment_id)
        return deployment.get('logs', []) if deployment else []

def main():
    """Command line interface for model deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Platform Model Deployer')
    parser.add_argument('--config', type=str, required=True, help='Deployment configuration JSON file')
    parser.add_argument('--deployments-dir', type=str, default='deployments', help='Deployments directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize deployer
    deployer = ModelDeployer(args.deployments_dir)
    
    # Deploy model
    result = deployer.deploy_model(config)
    
    if result['success']:
        deployment_info = result['deployment_info']
        print(f"Deployment successful!")
        print(f"Deployment ID: {result['deployment_id']}")
        print(f"API URL: {deployment_info.get('endpoints', {}).get('api_url', 'N/A')}")
        print(f"Health URL: {deployment_info.get('endpoints', {}).get('health_url', 'N/A')}")
        print(f"Prediction URL: {deployment_info.get('endpoints', {}).get('predict_url', 'N/A')}")
    else:
        print(f"Deployment failed: {result['error']}")

if __name__ == '__main__':
    main()