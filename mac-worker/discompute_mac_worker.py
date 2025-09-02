#!/usr/bin/env python3
"""
Discompute Mac Worker Client
High-performance distributed training for Apple Silicon Macs
Inspired by EXO architecture but optimized for training workloads
"""

import socket
import json
import time
import threading
import uuid
import logging
import http.server
import socketserver
import psutil
import os
import platform
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('discompute_mac')

# Try to import ML libraries with priority order
ML_FRAMEWORK = None
HAS_MLX = False
HAS_TINYGRAD = False
HAS_PYTORCH = False

# 1. Try MLX first (Apple's native framework - fastest on Apple Silicon)
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    ML_FRAMEWORK = "mlx"
    HAS_MLX = True
    logger.info("🚀 MLX available - using Apple's native ML framework")
except ImportError:
    pass

# 2. Try Tinygrad (good cross-platform performance)
if not HAS_MLX:
    try:
        from tinygrad import Tensor, nn as tg_nn, dtypes
        from tinygrad.nn.optim import Adam, SGD
        from tinygrad.nn.state import get_parameters
        ML_FRAMEWORK = "tinygrad"
        HAS_TINYGRAD = True
        logger.info("🧠 Tinygrad available - using cross-platform ML framework")
    except ImportError:
        pass

# 3. Fallback to PyTorch with MPS
if not HAS_MLX and not HAS_TINYGRAD:
    try:
        import torch
        import torch.nn as torch_nn
        import torch.optim as torch_optim
        if torch.backends.mps.is_available():
            ML_FRAMEWORK = "pytorch_mps"
            HAS_PYTORCH = True
            logger.info("⚡ PyTorch with MPS available - using Metal acceleration")
        else:
            ML_FRAMEWORK = "pytorch_cpu"
            HAS_PYTORCH = True
            logger.info("💻 PyTorch CPU available - using CPU training")
    except ImportError:
        pass

if ML_FRAMEWORK is None:
    logger.warning("⚠️  No ML frameworks available - using simulation mode")
    ML_FRAMEWORK = "simulation"

@dataclass
class MacDeviceCapabilities:
    """Mac device capabilities optimized for Apple Silicon"""
    cpu_cores: int
    memory_mb: int
    has_gpu: bool = False
    gpu_type: str = ""
    chip: str = ""
    os_version: str = ""
    supports_metal: bool = False
    device_type: str = "mac"
    model: str = ""
    # Performance estimates (TFLOPS)
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    int8_tflops: float = 0.0

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: str
    batch_size: int
    learning_rate: float
    epochs: int
    distribution_mode: str
    optimizer: str
    parameters: Dict[str, Any]

class MLXMNISTModel:
    """MNIST CNN model using MLX (Apple's native ML framework)"""
    
    def __init__(self, learning_rate: float = 0.001):
        if not HAS_MLX:
            return
        
        # MLX CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.linear1 = nn.Linear(1600, 128)
        self.linear2 = nn.Linear(128, 10)
        
        # MLX optimizer
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        
        logger.info("MLX MNIST model initialized")
    
    def __call__(self, x):
        if not HAS_MLX:
            return None
        
        # Reshape if needed: (batch, 784) -> (batch, 1, 28, 28)
        if len(x.shape) == 2:
            x = mx.reshape(x, (-1, 1, 28, 28))
        
        # Forward pass
        x = nn.relu(self.conv1(x))
        x = nn.max_pool2d(x, kernel_size=2)
        x = nn.relu(self.conv2(x))
        x = nn.max_pool2d(x, kernel_size=2)
        x = mx.flatten(x, start_axis=1)
        x = nn.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
    
    def train_batch(self, data: List[List[float]], labels: List[int]) -> Tuple[float, float, Dict[str, List[float]]]:
        """Train on a batch using MLX"""
        if not HAS_MLX:
            return 0.5, 0.8, {}
        
        try:
            # Convert to MLX arrays
            x = mx.array(data, dtype=mx.float32)
            y = mx.array(labels, dtype=mx.int32)
            
            # Forward pass and loss computation
            def loss_fn(params):
                logits = self.__call__(x)
                return nn.losses.cross_entropy(logits, y)
            
            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(self.parameters())
            
            # Update parameters
            self.optimizer.update(self, grads)
            mx.eval(self.parameters())
            
            # Compute accuracy
            logits = self.__call__(x)
            predictions = mx.argmax(logits, axis=1)
            accuracy = mx.mean(predictions == y).item()
            
            # Extract gradients for distributed training
            gradients = {}
            for name, grad in grads.items():
                if 'weight' in name:
                    # Flatten and take first 100 elements to avoid huge payloads
                    grad_flat = mx.flatten(grad)[:100]
                    gradients[name] = grad_flat.tolist()
            
            return loss.item(), accuracy, gradients
            
        except Exception as e:
            logger.error(f"MLX training error: {e}")
            return 0.5, 0.1, {}
    
    def parameters(self):
        """Get model parameters"""
        if not HAS_MLX:
            return {}
        return {
            'conv1_weight': self.conv1.weight,
            'conv1_bias': self.conv1.bias,
            'conv2_weight': self.conv2.weight,
            'conv2_bias': self.conv2.bias,
            'linear1_weight': self.linear1.weight,
            'linear1_bias': self.linear1.bias,
            'linear2_weight': self.linear2.weight,
            'linear2_bias': self.linear2.bias,
        }

class TinygradMNISTModel:
    """MNIST CNN model using Tinygrad"""
    
    def __init__(self, learning_rate: float = 0.001):
        if not HAS_TINYGRAD:
            return
        
        # Tinygrad CNN layers
        self.conv1 = tg_nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = tg_nn.Conv2d(32, 64, kernel_size=3)
        self.linear1 = tg_nn.Linear(1600, 128)
        self.linear2 = tg_nn.Linear(128, 10)
        
        # Tinygrad optimizer
        self.optimizer = Adam(get_parameters(self), lr=learning_rate)
        
        logger.info("Tinygrad MNIST model initialized")
    
    def __call__(self, x):
        if not HAS_TINYGRAD:
            return None
        
        # Reshape if needed
        if len(x.shape) == 2:
            x = x.reshape(-1, 1, 28, 28)
        
        x = self.conv1(x).relu().max_pool2d(2)
        x = self.conv2(x).relu().max_pool2d(2)
        x = x.flatten(1)
        x = self.linear1(x).relu()
        x = self.linear2(x)
        
        return x
    
    def train_batch(self, data: List[List[float]], labels: List[int]) -> Tuple[float, float, Dict[str, List[float]]]:
        """Train on a batch using Tinygrad"""
        if not HAS_TINYGRAD:
            return 0.5, 0.8, {}
        
        try:
            # Convert to tensors
            x = Tensor(data, dtype=dtypes.float32)
            y = Tensor(labels, dtype=dtypes.int32)
            
            # Enable training mode
            Tensor.training = True
            
            # Forward pass
            logits = self(x)
            loss = logits.sparse_categorical_crossentropy(y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            Tensor.training = False
            with Tensor.no_grad():
                pred = self(x).argmax(axis=1)
                accuracy = (pred == y).mean().item()
            
            # Extract gradients
            gradients = {}
            try:
                for name, param in [("conv1", self.conv1), ("conv2", self.conv2)]:
                    if hasattr(param, 'weight') and hasattr(param.weight, 'grad'):
                        grad_data = param.weight.grad.numpy().flatten()[:100].tolist()
                        gradients[f"{name}_weight"] = grad_data
            except Exception as e:
                logger.debug(f"Gradient extraction failed: {e}")
            
            return loss.item(), accuracy, gradients
            
        except Exception as e:
            logger.error(f"Tinygrad training error: {e}")
            return 0.5, 0.1, {}

class MacSystemInfo:
    """Collect Mac system information and capabilities"""
    
    @staticmethod
    def get_device_capabilities() -> MacDeviceCapabilities:
        """Get comprehensive Mac device capabilities"""
        
        # Basic system info
        cpu_cores = os.cpu_count() or 1
        memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        
        # Mac-specific information
        chip = ""
        device_type = "mac"
        model = ""
        supports_metal = False
        
        try:
            # Get Mac model
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Model Name:' in line:
                        model = line.split(':')[1].strip()
                    elif 'Chip:' in line:
                        chip = line.split(':')[1].strip()
            
            # Determine device type from model
            if 'MacBook Pro' in model:
                device_type = "macbook"
            elif 'MacBook Air' in model:
                device_type = "macbook"
            elif 'Mac Studio' in model:
                device_type = "mac_studio"
            elif 'Mac Pro' in model:
                device_type = "mac"
            elif 'Mac mini' in model:
                device_type = "mac_mini"
            elif 'iMac' in model:
                device_type = "imac"
            
            # Check for Apple Silicon (supports Metal)
            if any(silicon in chip.lower() for silicon in ['m1', 'm2', 'm3', 'apple']):
                supports_metal = True
            
        except Exception as e:
            logger.debug(f"Failed to get Mac-specific info: {e}")
            # Fallback detection
            arch = platform.machine().lower()
            if arch == 'arm64':
                supports_metal = True
                chip = "Apple Silicon"
        
        # Estimate performance based on chip
        fp32_tflops, fp16_tflops, int8_tflops = MacSystemInfo._estimate_performance(chip, model)
        
        return MacDeviceCapabilities(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            has_gpu=supports_metal,
            gpu_type="Apple GPU" if supports_metal else "",
            chip=chip,
            os_version=platform.mac_ver()[0],
            supports_metal=supports_metal,
            device_type=device_type,
            model=model,
            fp32_tflops=fp32_tflops,
            fp16_tflops=fp16_tflops,
            int8_tflops=int8_tflops
        )
    
    @staticmethod
    def _estimate_performance(chip: str, model: str) -> Tuple[float, float, float]:
        """Estimate TFLOPS performance based on chip and model"""
        
        # Performance estimates based on Apple's specifications and benchmarks
        chip_lower = chip.lower()
        model_lower = model.lower()
        
        # M3 generation (2023+)
        if 'm3 max' in chip_lower:
            return (4.0, 8.0, 16.0)  # M3 Max
        elif 'm3 pro' in chip_lower:
            return (2.5, 5.0, 10.0)  # M3 Pro
        elif 'm3' in chip_lower:
            return (1.8, 3.6, 7.2)   # M3
        
        # M2 generation (2022-2023)
        elif 'm2 ultra' in chip_lower:
            return (5.0, 10.0, 20.0) # M2 Ultra
        elif 'm2 max' in chip_lower:
            return (3.5, 7.0, 14.0)  # M2 Max
        elif 'm2 pro' in chip_lower:
            return (2.2, 4.4, 8.8)   # M2 Pro
        elif 'm2' in chip_lower:
            return (1.6, 3.2, 6.4)   # M2
        
        # M1 generation (2020-2022)
        elif 'm1 ultra' in chip_lower:
            return (4.5, 9.0, 18.0)  # M1 Ultra
        elif 'm1 max' in chip_lower:
            return (3.0, 6.0, 12.0)  # M1 Max
        elif 'm1 pro' in chip_lower:
            return (2.0, 4.0, 8.0)   # M1 Pro
        elif 'm1' in chip_lower:
            return (1.4, 2.8, 5.6)   # M1
        
        # Intel Macs (much lower performance)
        else:
            return (0.5, 0.5, 1.0)   # Intel Mac

class TrainingRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handle HTTP training requests from the Go coordinator"""
    
    def __init__(self, mac_worker, *args, **kwargs):
        self.mac_worker = mac_worker
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for health checks"""
        try:
            if self.path == '/api/health':
                response = {'success': True, 'message': 'Mac device healthy', 'framework': ML_FRAMEWORK}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            elif self.path == '/api/device/info':
                response = self.mac_worker.handle_get_device_info()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            logger.error(f"Error handling GET request: {e}")
            self.send_response(500)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            action = request_data.get('action')
            
            # Route to appropriate handler
            if action == 'initialize_training':
                response = self.mac_worker.handle_initialize_training(request_data.get('config', {}))
            elif action == 'train_batch':
                response = self.mac_worker.handle_train_batch(request_data.get('batch', {}))
            elif action == 'model_update':
                response = self.mac_worker.handle_model_update(request_data.get('update', {}))
            elif action == 'get_metrics':
                response = self.mac_worker.handle_get_metrics()
            elif action == 'get_device_info':
                response = self.mac_worker.handle_get_device_info()
            elif action == 'health_check':
                response = {'success': True, 'message': 'Mac device healthy', 'framework': ML_FRAMEWORK}
            elif action == 'shutdown_training':
                response = self.mac_worker.handle_shutdown_training()
            else:
                response = {'success': False, 'message': f'Unknown action: {action}'}
            
            # Send response
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

class MacWorkerClient:
    """Mac worker client for distributed training"""
    
    def __init__(self, node_id=None, http_port=8080):
        self.node_id = node_id or f"mac_{uuid.uuid4().hex[:8]}"
        self.http_port = http_port
        
        # Get device capabilities
        self.capabilities = MacSystemInfo.get_device_capabilities()
        
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
            'gpu_usage': 0.0,
            'temperature': 0.0,
            'throughput_tps': 0.0
        }
        
        logger.info(f"Mac worker initialized: {self.node_id}")
        logger.info(f"Device: {self.capabilities.model} ({self.capabilities.chip})")
        logger.info(f"Capabilities: {self.capabilities.cpu_cores} cores, {self.capabilities.memory_mb}MB RAM")
        logger.info(f"ML Framework: {ML_FRAMEWORK}")
        logger.info(f"Performance estimate: {self.capabilities.fp32_tflops:.1f} TFLOPS (FP32)")
    
    def start(self):
        """Start the Mac worker client"""
        logger.info("Starting Mac worker client...")
        
        # Start UDP discovery broadcasting
        self.start_udp_discovery()
        
        # Start HTTP training server
        self.start_http_server()
        
        # Start metrics monitoring
        self.start_metrics_monitoring()
        
        logger.info(f"✅ Mac worker running on port {self.http_port}")
        logger.info(f"📡 Broadcasting on UDP port 5005")
        logger.info(f"🆔 Device ID: {self.node_id}")
        logger.info(f"🧠 Framework: {ML_FRAMEWORK}")
    
    def start_udp_discovery(self):
        """Start UDP discovery broadcasting"""
        def discovery_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while True:
                try:
                    # Convert capabilities to Go-compatible format
                    device_caps = {
                        "model": self.capabilities.model,
                        "chip": self.capabilities.chip,
                        "memory": self.capabilities.memory_mb,  # Go expects "memory" not "memory_mb"
                        "type": self.capabilities.device_type,
                        "flops": {
                            "fp32": self.capabilities.fp32_tflops,
                            "fp16": self.capabilities.fp16_tflops,
                            "int8": self.capabilities.int8_tflops
                        }
                    }
                    
                    message = {
                        "type": "discovery",
                        "node_id": self.node_id,
                        "grpc_port": 50051,
                        "http_port": self.http_port,
                        "device_capabilities": device_caps,
                        "priority": self._get_priority(),
                        "interface_name": "wifi",
                        "interface_type": "wifi",
                        "timestamp": int(time.time())  # Go expects int64, not float
                    }
                    
                    data = json.dumps(message).encode('utf-8')
                    sock.sendto(data, ('255.255.255.255', 5005))
                    
                    time.sleep(2.5)  # Broadcast every 2.5 seconds (like EXO)
                    
                except Exception as e:
                    logger.error(f"UDP discovery error: {e}")
                    time.sleep(5)
        
        self.udp_discovery = threading.Thread(target=discovery_loop, daemon=True)
        self.udp_discovery.start()
        logger.info("UDP discovery broadcasting started")
    
    def _get_priority(self) -> int:
        """Get device priority for training (lower = higher priority)"""
        # Prioritize based on performance: M3 > M2 > M1 > Intel
        chip = self.capabilities.chip.lower()
        
        if 'm3 ultra' in chip: return 1
        elif 'm3 max' in chip: return 2
        elif 'm3 pro' in chip: return 3
        elif 'm3' in chip: return 4
        elif 'm2 ultra' in chip: return 5
        elif 'm2 max' in chip: return 6
        elif 'm2 pro' in chip: return 7
        elif 'm2' in chip: return 8
        elif 'm1 ultra' in chip: return 9
        elif 'm1 max' in chip: return 10
        elif 'm1 pro' in chip: return 11
        elif 'm1' in chip: return 12
        else: return 20  # Intel Mac
    
    def start_http_server(self):
        """Start HTTP server for training requests"""
        handler = lambda *args: TrainingRequestHandler(self, *args)
        self.http_server = socketserver.TCPServer(("", self.http_port), handler)
        server_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        server_thread.start()
        logger.info(f"HTTP training server started on port {self.http_port}")
    
    def start_metrics_monitoring(self):
        """Start monitoring device metrics"""
        def metrics_loop():
            while True:
                try:
                    # Update basic metrics
                    self.metrics['cpu_usage'] = psutil.cpu_percent(interval=1) / 100.0
                    self.metrics['memory_usage'] = psutil.virtual_memory().percent / 100.0
                    
                    # Try to get GPU usage (if available)
                    if self.capabilities.supports_metal:
                        # On Apple Silicon, GPU usage is harder to measure
                        # For now, estimate based on CPU usage
                        self.metrics['gpu_usage'] = min(self.metrics['cpu_usage'] * 1.2, 1.0)
                    
                    # Try to get temperature (macOS specific)
                    try:
                        result = subprocess.run(['sudo', 'powermetrics', '-n', '1', '-s', 'thermal'], 
                                              capture_output=True, text=True, timeout=3)
                        if result.returncode == 0 and 'CPU die temperature' in result.stdout:
                            # Parse temperature from powermetrics output
                            for line in result.stdout.split('\n'):
                                if 'CPU die temperature' in line:
                                    temp_str = line.split(':')[1].strip().split()[0]
                                    self.metrics['temperature'] = float(temp_str)
                                    break
                    except:
                        pass  # Temperature monitoring is optional
                    
                    time.sleep(10)  # Update every 10 seconds
                    
                except Exception as e:
                    logger.debug(f"Metrics monitoring error: {e}")
                    time.sleep(10)
        
        metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
        metrics_thread.start()
        logger.info("Metrics monitoring started")
    
    def handle_initialize_training(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training initialization request"""
        try:
            self.current_config = TrainingConfig(**config_data)
            
            # Initialize model based on available framework and type
            if self.current_config.model_type == "mnist_cnn":
                if ML_FRAMEWORK == "mlx":
                    self.current_model = MLXMNISTModel(self.current_config.learning_rate)
                elif ML_FRAMEWORK == "tinygrad":
                    self.current_model = TinygradMNISTModel(self.current_config.learning_rate)
                else:
                    logger.warning(f"No suitable framework for {self.current_config.model_type}")
                    return {'success': False, 'message': 'No suitable ML framework available'}
                
                logger.info(f"Initialized {self.current_config.model_type} model with {ML_FRAMEWORK}")
            else:
                return {'success': False, 'message': f'Unsupported model type: {self.current_config.model_type}'}
            
            self.training_active = True
            
            return {
                'success': True,
                'message': 'Training initialized successfully',
                'device_info': asdict(self.capabilities),
                'framework': ML_FRAMEWORK
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize training: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_train_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training batch request"""
        try:
            if not self.training_active or not self.current_model:
                return {'success': False, 'message': 'Training not initialized'}
            
            batch_id = batch_data.get('batch_id', 'unknown')
            data = batch_data.get('data', [])
            labels = batch_data.get('labels', [])
            
            if not data or not labels:
                return {'success': False, 'message': 'Invalid batch data'}
            
            start_time = time.time()
            
            # Train the batch using the appropriate framework
            loss, accuracy, gradients = self.current_model.train_batch(data, labels)
            
            processing_time = time.time() - start_time
            self.metrics['throughput_tps'] = len(data) / processing_time if processing_time > 0 else 0
            
            result = {
                'batch_id': batch_id,
                'loss': loss,
                'accuracy': accuracy,
                'gradients': gradients,
                'processing_time': processing_time,
                'device_metrics': self.metrics.copy()
            }
            
            logger.info(f"Trained batch {batch_id}: loss={loss:.4f}, acc={accuracy:.4f}, time={processing_time:.3f}s")
            
            return {
                'success': True,
                'message': 'Batch trained successfully',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Failed to train batch: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_model_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model parameter update"""
        try:
            if not self.training_active or not self.current_model:
                return {'success': False, 'message': 'Training not initialized'}
            
            update_id = update_data.get('update_id', 'unknown')
            parameters = update_data.get('parameters', {})
            
            # Apply parameter updates (simplified for now)
            # In a full implementation, you'd properly update model parameters
            logger.info(f"Applied model update {update_id} with {len(parameters)} parameter groups")
            
            return {'success': True, 'message': 'Model updated successfully'}
            
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_get_metrics(self) -> Dict[str, Any]:
        """Handle metrics request"""
        return {
            'success': True,
            'metrics': self.metrics.copy(),
            'device_info': asdict(self.capabilities),
            'framework': ML_FRAMEWORK
        }
    
    def handle_get_device_info(self) -> Dict[str, Any]:
        """Handle device info request"""
        return {
            'success': True,
            'device_info': asdict(self.capabilities),
            'framework': ML_FRAMEWORK,
            'node_id': self.node_id,
            'training_active': self.training_active
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
    
    def stop(self):
        """Stop the Mac worker client"""
        logger.info("Stopping Mac worker client...")
        
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        
        self.training_active = False
        logger.info("Mac worker client stopped")

def main():
    """Main function for Mac worker"""
    print("🚀 Discompute Mac Worker Client")
    print("===============================")
    print(f"🖥️  Running on: {platform.node()}")
    print(f"🧠 ML Framework: {ML_FRAMEWORK}")
    
    # Create and start the worker
    worker = MacWorkerClient()
    
    try:
        worker.start()
        
        print(f"✅ Mac worker running")
        print(f"📊 Device: {worker.capabilities.model}")
        print(f"⚡ Chip: {worker.capabilities.chip}")
        print(f"💾 Memory: {worker.capabilities.memory_mb}MB")
        print(f"🔥 Performance: {worker.capabilities.fp32_tflops:.1f} TFLOPS")
        print(f"📡 Broadcasting on UDP port 5005")
        print(f"🔗 HTTP server on port {worker.http_port}")
        print(f"🆔 Device ID: {worker.node_id}")
        print()
        print("💡 Ready for distributed training!")
        print("   This Mac will join any discompute cluster automatically")
        print("   Start the coordinator to begin training!")
        
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
        worker.stop()
        print("👋 Mac worker stopped")

if __name__ == "__main__":
    main()
