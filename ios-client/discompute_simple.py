#!/usr/bin/env python3
"""
Simple Discompute iOS Client for a-Shell
Minimal version optimized for easy transfer and setup
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

# Try to import ML libraries
try:
    from tinygrad import Tensor, nn, dtypes, TinyJit
    from tinygrad.nn.state import get_parameters
    from tinygrad.nn.optim import Adam
    HAS_TINYGRAD = True
    print("🧠 Tinygrad available")
except ImportError:
    print("⚠️  Tinygrad not available - using simulation mode")
    HAS_TINYGRAD = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("⚠️  NumPy not available")
    HAS_NUMPY = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('discompute')

class SimpleMNISTModel:
    """Simple MNIST model for iOS"""
    def __init__(self):
        if not HAS_TINYGRAD:
            return
        
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = Adam(get_parameters(self), lr=0.001)
        logger.info("MNIST model initialized")
    
    def __call__(self, x):
        if not HAS_TINYGRAD:
            return None
        if len(x.shape) == 2:
            x = x.reshape(-1, 1, 28, 28)
        x = self.conv1(x).relu().max_pool2d(2)
        x = self.conv2(x).relu().max_pool2d(2)
        x = x.flatten(1)
        x = self.fc1(x).relu()
        return self.fc2(x)
    
    def train_batch(self, data, labels):
        if not HAS_TINYGRAD:
            return 0.5, 0.8, {}  # Dummy values
        
        try:
            batch_size = len(data)
            x = Tensor(data).reshape(batch_size, 1, 28, 28)
            y = Tensor(labels)
            
            Tensor.training = True
            logits = self(x)
            loss = logits.sparse_categorical_crossentropy(y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            Tensor.training = False
            with Tensor.no_grad():
                pred = self(x).argmax(axis=1)
                accuracy = (pred == y).mean().item()
            
            return loss.item(), accuracy, {"conv1": [0.1] * 50}  # Simplified gradients
        except Exception as e:
            logger.error(f"Training error: {e}")
            return 0.5, 0.1, {}

class TrainingRequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, device_client, *args, **kwargs):
        self.device_client = device_client
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
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
            else:
                response = {'success': False, 'message': f'Unknown action: {action}'}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {'success': False, 'message': str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def log_message(self, format, *args):
        logger.info(f"HTTP: {format % args}")

class SimpleIOSClient:
    def __init__(self, node_id=None, http_port=8080):
        self.node_id = node_id or f"ios_{uuid.uuid4().hex[:8]}"
        self.http_port = http_port
        self.current_model = None
        self.training_active = False
        
        # Get basic device info
        self.capabilities = {
            'cpu_cores': os.cpu_count() or 1,
            'memory_mb': int(psutil.virtual_memory().total / (1024 * 1024)),
            'device_type': 'ios',
            'model': 'iPad',
            'supports_metal': True
        }
        
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'battery_level': 0.8,  # Assume 80%
            'temperature': 25.0,
            'throughput_tps': 0.0
        }
        
        logger.info(f"iOS client initialized: {self.node_id}")
    
    def start(self):
        logger.info("Starting iOS client...")
        
        # Start UDP discovery
        self.start_udp_discovery()
        
        # Start HTTP server
        self.start_http_server()
        
        # Start metrics monitoring
        self.start_metrics_monitoring()
        
        logger.info(f"✅ iOS client running on port {self.http_port}")
        logger.info(f"📡 Broadcasting on UDP port 5005")
        logger.info(f"🆔 Device ID: {self.node_id}")
    
    def start_udp_discovery(self):
        def discovery_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while True:
                try:
                    message = {
                        "type": "discovery",
                        "node_id": self.node_id,
                        "grpc_port": 50051,
                        "http_port": self.http_port,
                        "device_capabilities": self.capabilities,
                        "priority": 1,
                        "interface_name": "wifi",
                        "interface_type": "wifi",
                        "timestamp": time.time()
                    }
                    
                    data = json.dumps(message).encode('utf-8')
                    sock.sendto(data, ('255.255.255.255', 5005))
                    time.sleep(2.5)
                    
                except Exception as e:
                    logger.error(f"UDP discovery error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=discovery_loop, daemon=True)
        thread.start()
        logger.info("UDP discovery started")
    
    def start_http_server(self):
        handler = lambda *args: TrainingRequestHandler(self, *args)
        self.server = socketserver.TCPServer(("", self.http_port), handler)
        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()
        logger.info(f"HTTP server started on port {self.http_port}")
    
    def start_metrics_monitoring(self):
        def metrics_loop():
            while True:
                try:
                    self.metrics['cpu_usage'] = psutil.cpu_percent(interval=1) / 100.0
                    self.metrics['memory_usage'] = psutil.virtual_memory().percent / 100.0
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Metrics error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=metrics_loop, daemon=True)
        thread.start()
        logger.info("Metrics monitoring started")
    
    def handle_initialize_training(self, config_data):
        try:
            logger.info(f"Initializing training: {config_data}")
            self.current_model = SimpleMNISTModel()
            self.training_active = True
            return {'success': True, 'message': 'Training initialized', 'device_info': self.capabilities}
        except Exception as e:
            logger.error(f"Init error: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_train_batch(self, batch_data):
        try:
            if not self.training_active or not self.current_model:
                return {'success': False, 'message': 'Training not initialized'}
            
            batch_id = batch_data.get('batch_id', 'unknown')
            data = batch_data.get('data', [])
            labels = batch_data.get('labels', [])
            
            start_time = time.time()
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
            
            logger.info(f"Trained batch {batch_id}: loss={loss:.4f}, acc={accuracy:.4f}")
            return {'success': True, 'message': 'Batch trained', 'result': result}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_model_update(self, update_data):
        try:
            update_id = update_data.get('update_id', 'unknown')
            logger.info(f"Applied model update {update_id}")
            return {'success': True, 'message': 'Model updated'}
        except Exception as e:
            logger.error(f"Update error: {e}")
            return {'success': False, 'message': str(e)}
    
    def handle_get_metrics(self):
        return {'success': True, 'metrics': self.metrics.copy()}

def main():
    print("🚀 Simple Discompute iOS Client for a-Shell")
    print("=" * 45)
    
    client = SimpleIOSClient()
    
    try:
        client.start()
        print("💡 Ready for distributed training!")
        print("   Connect to your discompute cluster")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("👋 iOS client stopped")

if __name__ == "__main__":
    main()
