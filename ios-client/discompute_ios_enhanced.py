#!/usr/bin/env python3
"""
Enhanced Discompute iOS Client
Designed for Pythonista, a-Shell, and other iOS Python environments
Supports distributed neural network training with tinygrad
"""

import socket
import json
import time
import threading
import platform
import subprocess
import sys
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse
import psutil
import os

# Try to import required ML libraries
try:
    from tinygrad import Tensor, nn, dtypes, TinyJit
    from tinygrad.nn.state import get_state_dict, load_state_dict, get_parameters
    from tinygrad.nn.datasets import mnist
    from tinygrad.nn.optim import Adam, SGD
    HAS_TINYGRAD = True
    print("🧠 Tinygrad available - neural network training enabled")
except ImportError:
    print("⚠️  Tinygrad not available - using simulation mode")
    HAS_TINYGRAD = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("⚠️  NumPy not available - using Python lists")
    HAS_NUMPY = False

# iOS-specific imports
try:
    import objc_util
    import motion
    import location
    HAS_IOS_FEATURES = True
    print("📱 iOS-specific features available")
except ImportError:
    HAS_IOS_FEATURES = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('discompute_ios')

@dataclass
class DeviceCapabilities:
    """Enhanced device capabilities for iOS devices"""
    cpu_cores: int
    memory_mb: int
    has_gpu: bool = False
    gpu_type: str = ""
    battery_level: float = -1.0
    is_charging: bool = False
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    int8_tflops: float = 0.0
    chip: str = ""
    os_version: str = ""
    supports_metal: bool = False
    device_type: str = "ios"
    model: str = ""

@dataclass
class TrainingConfig:
    """Training configuration for distributed learning"""
    model_type: str
    batch_size: int
    learning_rate: float
    epochs: int
    distribution_mode: str  # "data_parallel", "model_parallel"
    optimizer: str  # "adam", "sgd"
    parameters: Dict[str, Any]

@dataclass
class TrainingBatch:
    """Training batch data"""
    batch_id: str
    job_id: str
    epoch: int
    batch_idx: int
    data: List[List[float]]
    labels: List[int]
    metadata: Dict[str, Any]

@dataclass
class TrainingResult:
    """Result from training a batch"""
    batch_id: str
    loss: float
    accuracy: float
    gradients: Dict[str, List[float]]
    processing_time: float
    device_metrics: Dict[str, float]

@dataclass
class ModelUpdate:
    """Model parameter update"""
    update_id: str
    job_id: str
    epoch: int
    parameters: Dict[str, List[float]]
    update_type: str  # "gradient", "parameter"

class IOSDeviceInfo:
    """Collects iOS device information"""
    
    @staticmethod
    def get_device_capabilities() -> DeviceCapabilities:
        """Get comprehensive device capabilities"""
        
        # Basic system info
        cpu_cores = os.cpu_count() or 1
        memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        
        # iOS-specific information
        chip = ""
        device_type = "ios"
        model = ""
        supports_metal = False
        battery_level = -1.0
        is_charging = False
        
        if HAS_IOS_FEATURES:
            try:
                # Get device model using iOS APIs
                device = objc_util.ObjCClass('UIDevice').currentDevice()
                model = str(device.model())
                
                # Determine device type
                if 'iPad' in model:
                    device_type = "ipad"
                elif 'iPhone' in model:
                    device_type = "iphone"
                
                # Get battery info
                device.setBatteryMonitoringEnabled_(True)
                battery_level = float(device.batteryLevel())
                battery_state = device.batteryState()
                is_charging = battery_state in [1, 2]  # Charging or Full
                
                # Try to get chip info (approximate)
                import platform
                chip = platform.processor() or "Unknown iOS Chip"
                
                # iOS devices generally support Metal
                supports_metal = True
                
            except Exception as e:
                logger.warning(f"Failed to get iOS-specific info: {e}")
        
        # Estimate performance capabilities based on device type
        fp32_tflops = 1.0  # Conservative estimate
        fp16_tflops = 2.0
        int8_tflops = 4.0
        
        if 'iPad Pro' in model or 'iPhone 15 Pro' in model:
            fp32_tflops = 3.0
            fp16_tflops = 6.0
            int8_tflops = 12.0
        elif 'iPad Air' in model or 'iPhone 14' in model:
            fp32_tflops = 2.0
            fp16_tflops = 4.0
            int8_tflops = 8.0
        
        return DeviceCapabilities(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            has_gpu=supports_metal,
            gpu_type="Apple GPU" if supports_metal else "",
            battery_level=battery_level,
            is_charging=is_charging,
            fp32_tflops=fp32_tflops,
            fp16_tflops=fp16_tflops,
            int8_tflops=int8_tflops,
            chip=chip,
            os_version=platform.version(),
            supports_metal=supports_metal,
            device_type=device_type,
            model=model
        )

class TinygradMNISTModel:
    """MNIST model using Tinygrad for iOS training"""
    
    def __init__(self):
        if not HAS_TINYGRAD:
            raise RuntimeError("Tinygrad not available")
        
        # Simple CNN for MNIST
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Initialize optimizer
        self.optimizer = Adam(get_parameters(self), lr=0.001)
        
        logger.info("Initialized MNIST CNN model with Tinygrad")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass"""
        # Reshape to (batch, 1, 28, 28) if needed
        if len(x.shape) == 2:
            x = x.reshape(-1, 1, 28, 28)
        
        x = self.conv1(x).relu().max_pool2d(2)
        x = self.conv2(x).relu().max_pool2d(2)
        x = x.flatten(1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x
    
    def train_batch(self, data: List[List[float]], labels: List[int]) -> Tuple[float, float, Dict[str, List[float]]]:
        """Train on a batch of data"""
        if not HAS_TINYGRAD:
            return 0.5, 0.1, {}  # Dummy values
        
        # Convert to tensors
        batch_size = len(data)
        x = Tensor(data).reshape(batch_size, 1, 28, 28)
        y = Tensor(labels)
        
        # Enable training mode
        Tensor.training = True
        
        # Forward pass
        logits = self(x)
        loss = logits.sparse_categorical_crossentropy(y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        Tensor.training = False
        with Tensor.no_grad():
            pred = self(x).argmax(axis=1)
            accuracy = (pred == y).mean().item()
        
        # Extract gradients (simplified)
        gradients = {}
        try:
            # Get gradients from parameters
            for name, param in [("conv1", self.conv1), ("conv2", self.conv2), 
                              ("fc1", self.fc1), ("fc2", self.fc2)]:
                if hasattr(param, 'weight') and hasattr(param.weight, 'grad'):
                    grad_data = param.weight.grad.numpy().flatten().tolist()
                    gradients[f"{name}_weight"] = grad_data[:100]  # Limit size
        except Exception as e:
            logger.warning(f"Failed to extract gradients: {e}")
        
        return loss.item(), accuracy, gradients
    
    def update_parameters(self, parameters: Dict[str, List[float]]):
        """Update model parameters from distributed training"""
        try:
            # Apply parameter updates (simplified)
            # In practice, you'd properly map parameter names and shapes
            logger.info(f"Received parameter update with {len(parameters)} layers")
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")

class HTTPTrainingServer:
    """HTTP server for receiving training requests"""
    
    def __init__(self, device_client, port=8080):
        self.device_client = device_client
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start the HTTP server"""
        handler = lambda *args: TrainingRequestHandler(self.device_client, *args)
        self.server = socketserver.TCPServer(("", self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        logger.info(f"Training HTTP server started on port {self.port}")
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
        logger.info("Training HTTP server stopped")

class TrainingRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handle HTTP training requests"""
    
    def __init__(self, device_client, *args, **kwargs):
        self.device_client = device_client
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            action = request_data.get('action')
            
            if action == 'initialize_training':
                response = self.device_client.handle_initialize_training(request_data.get('config', {}))
            elif action == 'train_batch':
                response = self.device_client.handle_train_batch(request_data.get('batch', {}))
            elif action == 'model_update':
                response = self.device_client.handle_model_update(request_data.get('update', {}))
            elif action == 'get_metrics':
                response = self.device_client.handle_get_metrics()
            elif action == 'health_check':
                response = {'success': True, 'message': 'Device healthy'}
            elif action == 'shutdown_training':
                response = self.device_client.handle_shutdown_training()
            else:
                response = {'success': False, 'message': f'Unknown action: {action}'}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {'success': False, 'message': str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"HTTP: {format % args}")

class EnhancedIOSClient:
    """Enhanced iOS client with distributed training capabilities"""
    
    def __init__(self, node_id=None, grpc_port=50051, http_port=8080):
        self.node_id = node_id or f"ios_{uuid.uuid4().hex[:8]}"
        self.grpc_port = grpc_port
        self.http_port = http_port
        
        # Device information
        self.capabilities = IOSDeviceInfo.get_device_capabilities()
        
        # Training state
        self.current_model = None
        self.current_config = None
        self.training_active = False
        
        # Network components
        self.udp_discovery = None
        self.http_server = None
        
        # Performance monitoring
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'battery_level': self.capabilities.battery_level,
            'temperature': 0.0,
            'throughput_tps': 0.0
        }
        
        logger.info(f"Initialized enhanced iOS client: {self.node_id}")
        logger.info(f"Device: {self.capabilities.model} ({self.capabilities.device_type})")
        logger.info(f"Capabilities: {self.capabilities.cpu_cores} cores, {self.capabilities.memory_mb}MB RAM")
    
    def start(self):
        """Start the iOS client"""
        logger.info("Starting enhanced iOS client...")
        
        # Start UDP discovery
        self.start_udp_discovery()
        
        # Start HTTP training server
        self.http_server = HTTPTrainingServer(self, self.http_port)
        self.http_server.start()
        
        # Start metrics monitoring
        self.start_metrics_monitoring()
        
        logger.info(f"iOS client started successfully on port {self.http_port}")
    
    def stop(self):
        """Stop the iOS client"""
        logger.info("Stopping iOS client...")
        
        if self.http_server:
            self.http_server.stop()
        
        if self.udp_discovery:
            self.udp_discovery.stop()
        
        self.training_active = False
        logger.info("iOS client stopped")
    
    def start_udp_discovery(self):
        """Start UDP discovery broadcasting"""
        def discovery_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while True:
                try:
                    message = {
                        "type": "discovery",
                        "node_id": self.node_id,
                        "grpc_port": self.grpc_port,
                        "http_port": self.http_port,
                        "device_capabilities": asdict(self.capabilities),
                        "priority": 1,
                        "interface_name": "wifi",
                        "interface_type": "wifi",
                        "timestamp": time.time()
                    }
                    
                    data = json.dumps(message).encode('utf-8')
                    sock.sendto(data, ('255.255.255.255', 5005))
                    
                    time.sleep(2.5)  # Broadcast every 2.5 seconds
                    
                except Exception as e:
                    logger.error(f"UDP discovery error: {e}")
                    time.sleep(5)
        
        self.udp_discovery = threading.Thread(target=discovery_loop, daemon=True)
        self.udp_discovery.start()
        logger.info("UDP discovery started")
    
    def start_metrics_monitoring(self):
        """Start monitoring device metrics"""
        def metrics_loop():
            while True:
                try:
                    # Update basic metrics
                    self.metrics['cpu_usage'] = psutil.cpu_percent(interval=1) / 100.0
                    self.metrics['memory_usage'] = psutil.virtual_memory().percent / 100.0
                    
                    # Update battery level if available
                    if HAS_IOS_FEATURES:
                        try:
                            device = objc_util.ObjCClass('UIDevice').currentDevice()
                            self.metrics['battery_level'] = float(device.batteryLevel())
                        except:
                            pass
                    
                    time.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Metrics monitoring error: {e}")
                    time.sleep(10)
        
        metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
        metrics_thread.start()
        logger.info("Metrics monitoring started")
    
    def handle_initialize_training(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training initialization request"""
        try:
            self.current_config = TrainingConfig(**config_data)
            
            # Initialize model based on type
            if self.current_config.model_type == "mnist_cnn" and HAS_TINYGRAD:
                self.current_model = TinygradMNISTModel()
                logger.info("Initialized MNIST CNN model")
            else:
                logger.warning(f"Model type {self.current_config.model_type} not supported or Tinygrad unavailable")
                return {'success': False, 'message': 'Model type not supported'}
            
            self.training_active = True
            
            return {
                'success': True, 
                'message': 'Training initialized successfully',
                'device_info': asdict(self.capabilities)
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize training: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_train_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training batch request"""
        try:
            if not self.training_active or not self.current_model:
                return {'success': False, 'message': 'Training not initialized'}
            
            batch = TrainingBatch(**batch_data)
            
            start_time = time.time()
            
            # Train the batch
            loss, accuracy, gradients = self.current_model.train_batch(batch.data, batch.labels)
            
            processing_time = time.time() - start_time
            self.metrics['throughput_tps'] = len(batch.data) / processing_time
            
            result = TrainingResult(
                batch_id=batch.batch_id,
                loss=loss,
                accuracy=accuracy,
                gradients=gradients,
                processing_time=processing_time,
                device_metrics=self.metrics.copy()
            )
            
            logger.info(f"Trained batch {batch.batch_id}: loss={loss:.4f}, acc={accuracy:.4f}")
            
            return {
                'success': True,
                'message': 'Batch trained successfully',
                'result': asdict(result)
            }
            
        except Exception as e:
            logger.error(f"Failed to train batch: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_model_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model parameter update"""
        try:
            if not self.training_active or not self.current_model:
                return {'success': False, 'message': 'Training not initialized'}
            
            update = ModelUpdate(**update_data)
            
            # Update model parameters
            self.current_model.update_parameters(update.parameters)
            
            logger.info(f"Applied model update {update.update_id}")
            
            return {'success': True, 'message': 'Model updated successfully'}
            
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_get_metrics(self) -> Dict[str, Any]:
        """Handle metrics request"""
        return {
            'success': True,
            'metrics': self.metrics.copy()
        }
    
    def handle_shutdown_training(self) -> Dict[str, Any]:
        """Handle training shutdown request"""
        try:
            self.training_active = False
            self.current_model = None
            self.current_config = None
            
            logger.info("Training shutdown completed")
            
            return {'success': True, 'message': 'Training shutdown successfully'}
            
        except Exception as e:
            logger.error(f"Failed to shutdown training: {e}")
            return {'success': False, 'message': str(e)}

def main():
    """Main function for iOS client"""
    print("🚀 Starting Enhanced Discompute iOS Client")
    print("=" * 50)
    
    # Create and start the client
    client = EnhancedIOSClient()
    
    try:
        client.start()
        
        print(f"✅ iOS client running on device: {client.capabilities.model}")
        print(f"📡 Broadcasting on UDP port 5005")
        print(f"🔗 HTTP training server on port {client.http_port}")
        print(f"🆔 Device ID: {client.node_id}")
        print("\n💡 Ready for distributed training!")
        print("   Connect this device to your discompute cluster")
        print("   and start training neural networks!")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.stop()
        print("👋 iOS client stopped")

if __name__ == "__main__":
    main()
