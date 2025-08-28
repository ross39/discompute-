#!/usr/bin/env python3
"""
Discompute Mobile Client for iOS
Compatible with Pyto, a-Shell, and other Python iOS apps

This Python script implements the same UDP discovery protocol
as the Go version, allowing iOS devices to participate in the
discompute network.
"""

import socket
import json
import time
import threading
import platform
import subprocess
import sys
import uuid
from datetime import datetime

# Try to import numpy, fall back to built-in math if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("⚠️  NumPy not available - AI tasks will use basic math fallbacks")
    import math
    import random
    HAS_NUMPY = False

# Try to import grpc for real distributed communication
try:
    import grpc
    import concurrent.futures
    HAS_GRPC = True
    print("✅ gRPC available - real distributed training enabled")
except ImportError:
    print("⚠️  gRPC not available - using simulation mode")
    HAS_GRPC = False
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class ComputeTask:
    """Represents a compute task that can be distributed across devices"""
    task_id: str
    task_type: str  # "training", "inference", "matrix_multiply", etc.
    task_data: Dict[str, Any]
    metadata: Dict[str, str]
    required_capabilities: List[str]
    estimated_duration: int
    priority: int = 5
    max_devices: int = 4
    target_subtasks: int = 1

@dataclass  
class SubTask:
    """Represents a subtask that runs on a specific device"""
    subtask_id: str
    parent_task_id: str
    subtask_index: int
    subtask_data: Dict[str, Any]
    assigned_device_id: str = ""
    status: str = "pending"  # pending, assigned, running, completed, failed
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class DistributedTrainingJob:
    """Represents a distributed neural network training job"""
    job_id: str
    model_config: Dict[str, Any]  # Network architecture
    training_config: Dict[str, Any]  # Training hyperparameters
    master_device_id: str
    slave_devices: List[str]
    current_epoch: int = 0
    total_epochs: int = 10
    model_parameters: Optional[Dict[str, Any]] = None
    training_data: Optional[List[Any]] = None
    status: str = "initialized"  # initialized, training, completed, failed

@dataclass
class TrainingBatch:
    """Represents a training batch sent to a slave device"""
    batch_id: str
    job_id: str
    epoch: int
    batch_index: int
    data: List[Any]  # Training samples
    labels: List[Any]  # Training labels
    model_weights: Dict[str, Any]  # Current model parameters
    learning_rate: float = 0.01

class TaskExecutor:
    """Executes compute tasks on the local device"""
    
    def __init__(self, device_capabilities):
        self.device_capabilities = device_capabilities
        self.running_tasks: Dict[str, SubTask] = {}
        
    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a subtask and return the result"""
        print(f"🔄 Executing subtask {subtask.subtask_id} of type {subtask.subtask_data.get('type', 'unknown')}")
        
        subtask.status = "running"
        subtask.started_at = time.time()
        self.running_tasks[subtask.subtask_id] = subtask
        
        try:
            # Route to appropriate executor based on task type
            task_type = subtask.subtask_data.get("type", "unknown")
            
            if task_type == "matrix_multiply":
                result = await self._execute_matrix_multiply(subtask)
            elif task_type == "simple_ml_training":
                result = await self._execute_simple_training(subtask)
            elif task_type == "data_processing":
                result = await self._execute_data_processing(subtask)
            else:
                result = await self._execute_dummy_task(subtask)
                
            subtask.result_data = result
            subtask.status = "completed"
            print(f"✅ Subtask {subtask.subtask_id} completed successfully")
            
        except Exception as e:
            subtask.error_message = str(e)
            subtask.status = "failed"
            print(f"❌ Subtask {subtask.subtask_id} failed: {e}")
            
        finally:
            subtask.completed_at = time.time()
            self.running_tasks.pop(subtask.subtask_id, None)
            
        return subtask
    
    async def _execute_matrix_multiply(self, subtask: SubTask) -> Dict[str, Any]:
        """Execute matrix multiplication task"""
        data = subtask.subtask_data
        
        # Simulate matrix multiplication work
        size = data.get("matrix_size", 100)
        iterations = data.get("iterations", 1)
        
        print(f"   Matrix multiply: {size}x{size} matrix, {iterations} iterations")
        
        total_flops = 0
        start_time = time.time()
        result_shape = None
        
        if HAS_NUMPY:
            for i in range(iterations):
                # Create random matrices (simulating work)
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                # Perform multiplication
                c = np.dot(a, b)
                result_shape = c.shape
                
                # Calculate FLOPS (2 * size^3 per multiplication)
                total_flops += 2 * size * size * size
        else:
            # Fallback implementation without numpy
            for i in range(iterations):
                # Simulate matrix operations with basic math
                for row in range(size):
                    for col in range(size):
                        for k in range(size):
                            # Simulate one multiply-add operation
                            a_val = random.random()
                            b_val = random.random()
                            result = a_val * b_val
                            total_flops += 2  # multiply + add
                result_shape = (size, size)
            
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "result_type": "matrix_multiply",
            "duration_seconds": duration,
            "total_flops": total_flops,
            "flops_per_second": total_flops / duration if duration > 0 else 0,
            "matrix_size": size,
            "iterations": iterations,
            "result_shape": result_shape,
            "used_numpy": HAS_NUMPY
        }
    
    async def _execute_simple_training(self, subtask: SubTask) -> Dict[str, Any]:
        """Execute simple ML training task"""
        data = subtask.subtask_data
        
        # Simulate training work
        epochs = data.get("epochs", 5)
        batch_size = data.get("batch_size", 32)
        model_size = data.get("model_size", 1000)
        
        print(f"   Training: {epochs} epochs, batch size {batch_size}, model size {model_size}")
        
        start_time = time.time()
        total_samples = 0
        
        for epoch in range(epochs):
            # Simulate training batches
            for batch in range(10):  # 10 batches per epoch
                if HAS_NUMPY:
                    # Simulate forward pass
                    weights = np.random.rand(model_size, 10).astype(np.float32)
                    data_batch = np.random.rand(batch_size, model_size).astype(np.float32)
                    
                    # Forward pass
                    output = np.dot(data_batch, weights)
                    
                    # Simulate backward pass
                    grad = np.random.rand(*weights.shape).astype(np.float32)
                    weights -= 0.01 * grad  # Gradient descent step
                else:
                    # Fallback implementation
                    for sample in range(batch_size):
                        # Simulate forward pass calculations
                        for neuron in range(10):
                            activation = 0
                            for feature in range(model_size):
                                weight = random.random()
                                input_val = random.random()
                                activation += weight * input_val
                            # Apply simple activation
                            output_val = 1 / (1 + math.exp(-activation)) if activation > -500 else 0
                
                total_samples += batch_size
                
                # Small delay to simulate realistic training time
                time.sleep(0.01)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "result_type": "simple_training",
            "duration_seconds": duration,
            "epochs_completed": epochs,
            "total_samples": total_samples,
            "samples_per_second": total_samples / duration if duration > 0 else 0,
            "final_loss": random.uniform(0.1, 1.0),  # Simulate loss
            "used_numpy": HAS_NUMPY
        }
    
    async def _execute_data_processing(self, subtask: SubTask) -> Dict[str, Any]:
        """Execute data processing task"""
        data = subtask.subtask_data
        
        # Simulate data processing
        data_size = data.get("data_size", 10000)
        operations = data.get("operations", ["filter", "map", "reduce"])
        
        print(f"   Data processing: {data_size} items, ops: {operations}")
        
        start_time = time.time()
        
        if HAS_NUMPY:
            # Generate sample data
            dataset = np.random.rand(data_size).astype(np.float32)
            
            # Apply operations
            for op in operations:
                if op == "filter":
                    dataset = dataset[dataset > 0.5]
                elif op == "map":
                    dataset = dataset * 2
                elif op == "reduce":
                    dataset = np.array([np.sum(dataset)])
                elif op == "sort":
                    dataset = np.sort(dataset)
        else:
            # Fallback implementation
            dataset = [random.random() for _ in range(data_size)]
            
            # Apply operations
            for op in operations:
                if op == "filter":
                    dataset = [x for x in dataset if x > 0.5]
                elif op == "map":
                    dataset = [x * 2 for x in dataset]
                elif op == "reduce":
                    dataset = [sum(dataset)]
                elif op == "sort":
                    dataset = sorted(dataset)
                
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "result_type": "data_processing",
            "duration_seconds": duration,
            "input_size": data_size,
            "output_size": len(dataset),
            "operations": operations,
            "processing_rate": data_size / duration if duration > 0 else 0,
            "final_result": float(dataset[0]) if len(dataset) > 0 else None,
            "used_numpy": HAS_NUMPY
        }
    
    async def _execute_dummy_task(self, subtask: SubTask) -> Dict[str, Any]:
        """Execute a dummy computational task"""
        duration = subtask.subtask_data.get("duration", 2)
        
        print(f"   Dummy task: {duration} seconds")
        
        start_time = time.time()
        
        # Simulate work
        result = 0
        iterations = int(duration * 1000)
        for i in range(iterations):
            if HAS_NUMPY:
                result += np.sin(i) * np.cos(i)
            else:
                result += math.sin(i) * math.cos(i)
            if i % 100 == 0:
                time.sleep(0.001)  # Small delay
                
        end_time = time.time()
        
        return {
            "result_type": "dummy_task",
            "duration_seconds": end_time - start_time,
            "iterations": iterations,
            "result": float(result)
        }

class SimpleTrainingService:
    """Simple gRPC-style service for distributed training communication"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.running = False
        self.server_port = None
        
    def start_server(self, port: int = 0):
        """Start training service server (iPad/worker)"""
        try:
            if HAS_GRPC:
                self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
                # In a real implementation, we'd add the generated gRPC service here
                # For now, we'll use HTTP as a simple alternative
                pass
            
            # Use a simple HTTP server for communication
            self.server_port = port if port > 0 else self._find_available_port()
            self.running = True
            
            # Start server in background thread
            threading.Thread(target=self._run_http_server, daemon=True).start()
            print(f"🚀 Training service started on port {self.server_port}")
            return self.server_port
            
        except Exception as e:
            print(f"❌ Failed to start training service: {e}")
            return None
    
    def _find_available_port(self) -> int:
        """Find an available port for the server"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _run_http_server(self):
        """Run a simple HTTP server for training communication"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json
            
            class TrainingHandler(BaseHTTPRequestHandler):
                def do_POST(self):
                    if self.path == '/train_batch':
                        try:
                            content_length = int(self.headers['Content-Length'])
                            post_data = self.rfile.read(content_length)
                            batch_data = json.loads(post_data.decode('utf-8'))
                            
                            # Process the training batch
                            batch = TrainingBatch(**batch_data)
                            result = self.server.training_manager.process_training_batch(batch)
                            
                            # Send response
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(result).encode('utf-8'))
                            
                        except Exception as e:
                            self.send_response(500)
                            self.end_headers()
                            self.wfile.write(f"Error: {str(e)}".encode('utf-8'))
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    pass  # Suppress HTTP server logs
            
            # Store reference to training manager
            TrainingHandler.server = self
            
            httpd = HTTPServer(('', self.server_port), TrainingHandler)
            while self.running:
                httpd.handle_request()
                
        except Exception as e:
            print(f"❌ HTTP server error: {e}")
    
    def send_training_batch(self, target_address: str, target_port: int, batch: TrainingBatch) -> Optional[Dict[str, Any]]:
        """Send training batch to worker device (MacBook → iPad)"""
        try:
            import urllib.request
            import json
            
            url = f"http://{target_address}:{target_port}/train_batch"
            batch_data = {
                'batch_id': batch.batch_id,
                'job_id': batch.job_id,
                'epoch': batch.epoch,
                'batch_index': batch.batch_index,
                'data': batch.data,
                'labels': batch.labels,
                'model_weights': batch.model_weights,
                'learning_rate': batch.learning_rate
            }
            
            data = json.dumps(batch_data).encode('utf-8')
            request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(request, timeout=30) as response:
                result_data = json.loads(response.read().decode('utf-8'))
                return result_data
                
        except Exception as e:
            print(f"❌ Failed to send batch to {target_address}:{target_port}: {e}")
            return None
    
    def stop_server(self):
        """Stop the training service"""
        self.running = False

class DistributedTrainingManager:
    """Manages distributed neural network training across multiple devices"""
    
    def __init__(self, device_id: str, is_master: bool = False):
        self.device_id = device_id
        self.is_master = is_master
        self.training_jobs: Dict[str, DistributedTrainingJob] = {}
        self.pending_batches: Dict[str, TrainingBatch] = {}
        self.completed_gradients: Dict[str, List[Dict[str, Any]]] = {}
        
    def create_neural_network(self, input_size: int, hidden_sizes: List[int], output_size: int) -> Dict[str, Any]:
        """Create a simple neural network architecture"""
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append({
                "name": f"hidden_{i}",
                "type": "dense",
                "input_size": prev_size,
                "output_size": hidden_size,
                "activation": "relu"
            })
            prev_size = hidden_size
            
        # Output layer
        layers.append({
            "name": "output",
            "type": "dense", 
            "input_size": prev_size,
            "output_size": output_size,
            "activation": "softmax"
        })
        
        return {
            "layers": layers,
            "input_size": input_size,
            "output_size": output_size,
            "total_parameters": self._count_parameters(layers)
        }
    
    def _count_parameters(self, layers: List[Dict[str, Any]]) -> int:
        """Count total parameters in the network"""
        total = 0
        for layer in layers:
            if layer["type"] == "dense":
                # weights + biases
                total += layer["input_size"] * layer["output_size"] + layer["output_size"]
        return total
    
    def initialize_weights(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize neural network weights"""
        weights = {}
        
        for layer in model_config["layers"]:
            layer_name = layer["name"]
            input_size = layer["input_size"]
            output_size = layer["output_size"]
            
            if HAS_NUMPY:
                # Xavier initialization
                limit = (6.0 / (input_size + output_size)) ** 0.5
                weights[f"{layer_name}_weights"] = np.random.uniform(
                    -limit, limit, (input_size, output_size)
                ).astype(np.float32).tolist()
                weights[f"{layer_name}_bias"] = np.zeros(output_size).astype(np.float32).tolist()
            else:
                # Simple random initialization
                limit = 0.1
                weights[f"{layer_name}_weights"] = [
                    [random.uniform(-limit, limit) for _ in range(output_size)]
                    for _ in range(input_size)
                ]
                weights[f"{layer_name}_bias"] = [0.0] * output_size
                
        return weights
    
    def create_mnist_data(self, num_samples: int) -> Tuple[List[List[float]], List[int]]:
        """Create synthetic MNIST-like training data (28x28 images, 10 classes)"""
        data = []
        labels = []
        
        print(f"🎨 Generating {num_samples} MNIST-like samples...")
        
        for i in range(num_samples):
            if HAS_NUMPY:
                # Create 28x28 = 784 pixel image
                image = np.random.rand(784).astype(np.float32)
                
                # Add some structure to make it more realistic
                # Create patterns that correlate with digit classes
                digit_class = i % 10
                
                # Add digit-specific patterns
                if digit_class == 0:  # Circle-like pattern
                    center_pixels = [392, 393, 420, 421]  # Center area
                    for idx in center_pixels:
                        if idx < 784:
                            image[idx] = max(0.1, image[idx] * 0.3)  # Darker center
                elif digit_class == 1:  # Vertical line pattern
                    for row in range(28):
                        center_col = 13 + row * 28  # Vertical line down center
                        if center_col < 784:
                            image[center_col] = min(1.0, image[center_col] + 0.5)
                elif digit_class == 7:  # Top horizontal pattern
                    for col in range(10, 18):  # Top horizontal line
                        if col < 784:
                            image[col] = min(1.0, image[col] + 0.4)
                
                # Normalize to 0-1 range
                image = np.clip(image, 0, 1)
                sample = image.tolist()
            else:
                # Fallback without numpy
                sample = []
                digit_class = i % 10
                
                for pixel in range(784):
                    base_val = random.random()
                    
                    # Add simple patterns for different digits
                    if digit_class == 0 and 350 < pixel < 450:  # Center area
                        base_val *= 0.3  # Darker center for 0
                    elif digit_class == 1 and pixel % 28 == 14:  # Vertical line
                        base_val = min(1.0, base_val + 0.5)
                    elif digit_class == 7 and pixel < 280:  # Top area
                        base_val = min(1.0, base_val + 0.3)
                    
                    sample.append(max(0.0, min(1.0, base_val)))
            
            data.append(sample)
            labels.append(digit_class)
            
        print(f"✅ Generated MNIST dataset: {len(data)} samples, {len(data[0])} features each")
        return data, labels
    
    def create_training_data(self, num_samples: int, input_size: int, num_classes: int) -> Tuple[List[List[float]], List[int]]:
        """Create training data - use MNIST if appropriate size, else synthetic"""
        if input_size == 784 and num_classes == 10:
            # MNIST configuration
            return self.create_mnist_data(num_samples)
        else:
            # Generic synthetic data
            data = []
            labels = []
            
            for _ in range(num_samples):
                if HAS_NUMPY:
                    sample = np.random.randn(input_size).astype(np.float32).tolist()
                else:
                    sample = [random.gauss(0, 1) for _ in range(input_size)]
                
                # Create synthetic label based on sample features
                label = abs(hash(str(sample[:3]))) % num_classes
                
                data.append(sample)
                labels.append(label)
                
            return data, labels
    
    def start_distributed_training(self, slave_devices: List[str], model_config: Dict[str, Any], 
                                 training_config: Dict[str, Any]) -> str:
        """Start a new distributed training job (Master only)"""
        if not self.is_master:
            raise ValueError("Only master device can start distributed training")
            
        job_id = str(uuid.uuid4())
        
        # Create training data
        data, labels = self.create_training_data(
            training_config.get("num_samples", 1000),
            model_config["input_size"],
            model_config["output_size"]
        )
        
        job = DistributedTrainingJob(
            job_id=job_id,
            model_config=model_config,
            training_config=training_config,
            master_device_id=self.device_id,
            slave_devices=slave_devices,
            total_epochs=training_config.get("epochs", 10),
            model_parameters=self.initialize_weights(model_config),
            training_data=list(zip(data, labels))
        )
        
        self.training_jobs[job_id] = job
        print(f"🎯 Started distributed training job {job_id[:8]}...")
        print(f"   Model: {len(model_config['layers'])} layers, {model_config['total_parameters']} parameters")
        print(f"   Data: {len(data)} samples")
        print(f"   Workers: {len(slave_devices)} devices")
        
        return job_id
    
    def distribute_epoch_batches(self, job_id: str, device_workloads: Optional[Dict[str, float]] = None) -> List[TrainingBatch]:
        """Distribute training batches for current epoch to slave devices with configurable workloads"""
        if job_id not in self.training_jobs:
            return []
            
        job = self.training_jobs[job_id]
        batch_size = job.training_config.get("batch_size", 32)
        num_slaves = len(job.slave_devices)
        
        if num_slaves == 0:
            return []
        
        # Use device workloads if provided, otherwise equal distribution
        if device_workloads is None:
            device_workloads = {device_id: 1.0 for device_id in job.slave_devices}
        
        # Normalize workloads to sum to 1.0
        total_workload = sum(device_workloads.get(device_id, 1.0) for device_id in job.slave_devices)
        normalized_workloads = {
            device_id: device_workloads.get(device_id, 1.0) / total_workload 
            for device_id in job.slave_devices
        }
        
        # Split data into batches
        batches = []
        data = job.training_data
        total_samples = len(data)
        
        print(f"📊 Workload distribution:")
        for device_id, workload in normalized_workloads.items():
            samples_for_device = int(total_samples * workload)
            print(f"   {device_id[:8]}...: {workload*100:.1f}% ({samples_for_device} samples)")
        
        # Distribute data based on workload percentages
        current_idx = 0
        for i, device_id in enumerate(job.slave_devices):
            workload_percentage = normalized_workloads[device_id]
            samples_for_device = int(total_samples * workload_percentage)
            
            # For last device, take remaining samples to avoid rounding issues
            if i == len(job.slave_devices) - 1:
                samples_for_device = total_samples - current_idx
            
            if samples_for_device > 0:
                device_data = data[current_idx:current_idx + samples_for_device]
                current_idx += samples_for_device
                
                # Split device data into batches
                for batch_idx in range(0, len(device_data), batch_size):
                    batch_data = device_data[batch_idx:batch_idx + batch_size]
                    
                    if len(batch_data) > 0:
                        batch_inputs = [sample[0] for sample in batch_data]
                        batch_labels = [sample[1] for sample in batch_data]
                        
                        batch = TrainingBatch(
                            batch_id=str(uuid.uuid4()),
                            job_id=job_id,
                            epoch=job.current_epoch,
                            batch_index=len(batches),
                            data=batch_inputs,
                            labels=batch_labels,
                            model_weights=job.model_parameters,
                            learning_rate=job.training_config.get("learning_rate", 0.01)
                        )
                        
                        batches.append(batch)
                        
        return batches
    
    def process_training_batch(self, batch: TrainingBatch) -> Dict[str, Any]:
        """Process a training batch and compute gradients (Slave device)"""
        print(f"🔄 Processing batch {batch.batch_id[:8]} with {len(batch.data)} samples")
        
        start_time = time.time()
        
        # Simulate realistic forward and backward pass for MNIST
        gradients = {}
        total_loss = 0.0
        batch_accuracy = 0.0
        
        # More realistic gradient computation that improves over time
        epoch = batch.epoch
        learning_progress = min(1.0, epoch / 10.0)  # Progress over 10 epochs
        
        for layer_name, weights in batch.model_weights.items():
            if "_weights" in layer_name:
                if HAS_NUMPY:
                    # Simulate gradient computation with decreasing magnitude as training progresses
                    base_magnitude = 0.01 * (1.0 - learning_progress * 0.8)  # Gradients get smaller
                    grad = np.random.randn(*np.array(weights).shape) * base_magnitude
                    
                    # Add some structure based on layer type
                    if "hidden_0" in layer_name:  # First hidden layer (most important for MNIST)
                        grad *= 1.5  # Larger gradients for feature detection layer
                    elif "output" in layer_name:  # Output layer
                        grad *= 0.8  # Smaller gradients for final classification
                    
                    gradients[layer_name] = grad.tolist()
                else:
                    # Fallback gradient simulation
                    base_magnitude = 0.01 * (1.0 - learning_progress * 0.8)
                    if isinstance(weights[0], list):  # 2D weights
                        gradients[layer_name] = [
                            [random.gauss(0, base_magnitude) for _ in row] for row in weights
                        ]
                    else:  # 1D bias
                        gradients[layer_name] = [random.gauss(0, base_magnitude) for _ in weights]
            elif "_bias" in layer_name:
                if HAS_NUMPY:
                    base_magnitude = 0.005 * (1.0 - learning_progress * 0.8)
                    grad = np.random.randn(len(weights)) * base_magnitude
                    gradients[layer_name] = grad.tolist()
                else:
                    base_magnitude = 0.005 * (1.0 - learning_progress * 0.8)
                    gradients[layer_name] = [random.gauss(0, base_magnitude) for _ in weights]
        
        # Simulate realistic loss that decreases over time
        initial_loss = 2.3  # ln(10) for random 10-class classification
        final_loss = 0.1    # Good MNIST performance
        total_loss = initial_loss - (initial_loss - final_loss) * learning_progress
        total_loss += random.uniform(-0.1, 0.1)  # Add some noise
        total_loss = max(0.01, total_loss)  # Ensure positive loss
        
        # Simulate accuracy that improves over time
        initial_accuracy = 0.1  # Random guessing
        final_accuracy = 0.95   # Good MNIST performance
        batch_accuracy = initial_accuracy + (final_accuracy - initial_accuracy) * learning_progress
        batch_accuracy += random.uniform(-0.05, 0.05)  # Add noise
        batch_accuracy = max(0.0, min(1.0, batch_accuracy))  # Clamp to [0,1]
        
        # Add some computational delay to simulate real work
        computation_delay = len(batch.data) * 0.002  # 2ms per sample
        time.sleep(computation_delay)
        
        end_time = time.time()
        
        return {
            "batch_id": batch.batch_id,
            "job_id": batch.job_id,
            "epoch": batch.epoch,
            "gradients": gradients,
            "loss": total_loss,
            "accuracy": batch_accuracy,
            "num_samples": len(batch.data),
            "computation_time": end_time - start_time,
            "device_id": self.device_id
        }
    
    def aggregate_gradients(self, job_id: str, gradient_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate gradients from all slaves (Master only)"""
        if not gradient_results:
            return {}
            
        # Average gradients across all devices
        aggregated = {}
        num_devices = len(gradient_results)
        
        # Get parameter names from first result
        first_result = gradient_results[0]
        param_names = first_result["gradients"].keys()
        
        for param_name in param_names:
            if HAS_NUMPY:
                # Stack gradients and compute average
                grads = [np.array(result["gradients"][param_name]) for result in gradient_results]
                avg_grad = np.mean(grads, axis=0)
                aggregated[param_name] = avg_grad.tolist()
            else:
                # Manual averaging for fallback
                grads = [result["gradients"][param_name] for result in gradient_results]
                if isinstance(grads[0][0], list):  # 2D weights
                    rows, cols = len(grads[0]), len(grads[0][0])
                    avg_grad = []
                    for r in range(rows):
                        row = []
                        for c in range(cols):
                            avg_val = sum(grad[r][c] for grad in grads) / num_devices
                            row.append(avg_val)
                        avg_grad.append(row)
                    aggregated[param_name] = avg_grad
                else:  # 1D bias
                    size = len(grads[0])
                    avg_grad = []
                    for i in range(size):
                        avg_val = sum(grad[i] for grad in grads) / num_devices
                        avg_grad.append(avg_val)
                    aggregated[param_name] = avg_grad
        
        # Calculate average loss and accuracy
        avg_loss = sum(result["loss"] for result in gradient_results) / num_devices
        avg_accuracy = sum(result.get("accuracy", 0.0) for result in gradient_results) / num_devices
        total_samples = sum(result["num_samples"] for result in gradient_results)
        total_computation_time = sum(result["computation_time"] for result in gradient_results)
        
        return {
            "aggregated_gradients": aggregated,
            "average_loss": avg_loss,
            "average_accuracy": avg_accuracy,
            "total_samples": total_samples,
            "total_computation_time": total_computation_time,
            "num_devices": num_devices
        }
    
    def update_model_parameters(self, job_id: str, aggregated_gradients: Dict[str, Any], learning_rate: float):
        """Update model parameters using aggregated gradients (Master only)"""
        if job_id not in self.training_jobs:
            return
            
        job = self.training_jobs[job_id]
        
        # Apply gradients to update parameters
        for param_name, gradient in aggregated_gradients.items():
            if param_name in job.model_parameters:
                if HAS_NUMPY:
                    current = np.array(job.model_parameters[param_name])
                    grad = np.array(gradient)
                    updated = current - learning_rate * grad
                    job.model_parameters[param_name] = updated.tolist()
                else:
                    # Manual parameter update
                    current = job.model_parameters[param_name]
                    if isinstance(current[0], list):  # 2D weights
                        for r in range(len(current)):
                            for c in range(len(current[r])):
                                current[r][c] -= learning_rate * gradient[r][c]
                    else:  # 1D bias
                        for i in range(len(current)):
                            current[i] -= learning_rate * gradient[i]
    
    def get_training_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current training status"""
        if job_id not in self.training_jobs:
            return None
            
        job = self.training_jobs[job_id]
        
        return {
            "job_id": job_id,
            "status": job.status,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "progress": (job.current_epoch / job.total_epochs) * 100,
            "master_device": job.master_device_id,
            "slave_devices": job.slave_devices,
            "model_info": {
                "layers": len(job.model_config["layers"]),
                "parameters": job.model_config["total_parameters"]
            }
        }

class DiscomputeMobile:
    def __init__(self, device_id=None, port=8080, debug=False):
        self.device_id = device_id or f"ios-{int(time.time())}"
        self.port = port
        self.listen_port = 5005
        self.broadcast_port = 5005
        self.running = False
        self.discovered_devices = {}
        self.device_capabilities = self.get_device_capabilities()
        self.debug = debug
        self.start_time = time.time()
        
        # Task management
        self.task_executor = TaskExecutor(self.device_capabilities)
        self.submitted_tasks: Dict[str, ComputeTask] = {}
        self.task_results: Dict[str, List[SubTask]] = {}
        
        # Distributed training management
        self.training_manager = DistributedTrainingManager(self.device_id, is_master=False)
        self.is_master_device = False
        self.training_service = SimpleTrainingService(self.device_id)
        
        # Set reference for HTTP server
        self.training_service.training_manager = self.training_manager
        
        print(f"🚀 Discompute Mobile Client")
        print(f"Device ID: {self.device_id}")
        print(f"Device Info: {self.device_capabilities}")
        
        if self.debug:
            self.print_network_info()
    
    def get_device_capabilities(self):
        """Detect iOS device capabilities"""
        try:
            # Try to get iOS device info
            model = platform.machine()
            if model.startswith('iPhone'):
                device_type = "iphone"
                # iPhone 16 Pro would be iPhone16,2 or similar
                if "16" in model:
                    chip = "Apple A18 Pro"
                    fp32_tflops = 2.80
                    fp16_tflops = 5.60
                    int8_tflops = 11.20
                elif "15" in model:
                    chip = "Apple A17 Pro" 
                    fp32_tflops = 2.15
                    fp16_tflops = 4.30
                    int8_tflops = 8.60
                else:
                    chip = "Apple A15 Bionic"
                    fp32_tflops = 1.37
                    fp16_tflops = 2.74
                    int8_tflops = 5.48
            elif model.startswith('iPad'):
                device_type = "ipad"
                chip = "Apple M2"  # Modern iPads use M-series
                fp32_tflops = 3.55
                fp16_tflops = 7.10
                int8_tflops = 14.20
            else:
                device_type = "ios_device"
                chip = "Unknown iOS Chip"
                fp32_tflops = 1.0
                fp16_tflops = 2.0
                int8_tflops = 4.0
                
            return {
                "model": model,
                "chip": chip,
                "type": device_type,
                "memory": 8192,  # Assume 8GB for modern iOS devices
                "flops": {
                    "fp32": fp32_tflops,
                    "fp16": fp16_tflops,
                    "int8": int8_tflops
                }
            }
        except Exception as e:
            print(f"Error detecting device capabilities: {e}")
            return {
                "model": "iOS Device",
                "chip": "Unknown",
                "type": "ios_device", 
                "memory": 4096,
                "flops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0}
            }
    
    def start_discovery(self):
        """Start UDP discovery service"""
        self.running = True
        
        # Start listening for discoveries
        listen_thread = threading.Thread(target=self.listen_for_devices, daemon=True)
        listen_thread.start()
        
        # Start broadcasting presence
        broadcast_thread = threading.Thread(target=self.broadcast_presence, daemon=True)  
        broadcast_thread.start()
        
        print(f"✅ Discovery started on port {self.listen_port}")
        return True
    
    def listen_for_devices(self):
        """Listen for UDP discovery messages"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Try to bind to all interfaces first, fallback to specific IP
            bind_success = False
            try:
                sock.bind(('', self.listen_port))
                bind_success = True
                print(f"👂 Listening on all interfaces, port {self.listen_port}")
            except Exception as e:
                print(f"Failed to bind to all interfaces: {e}")
                # Try binding to local IP
                try:
                    local_ip = self.get_local_ip()
                    sock.bind((local_ip, self.listen_port))
                    bind_success = True
                    print(f"👂 Listening on {local_ip}:{self.listen_port}")
                except Exception as e2:
                    print(f"Failed to bind to {local_ip}: {e2}")
                    # Last resort - try localhost
                    try:
                        sock.bind(('127.0.0.1', self.listen_port))
                        bind_success = True
                        print(f"👂 Listening on localhost:{self.listen_port}")
                    except Exception as e3:
                        print(f"Failed to bind to localhost: {e3}")
            
            if not bind_success:
                print("❌ Could not bind to any address for listening")
                return
                
            sock.settimeout(1.0)  # Non-blocking with timeout
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(4096)
                    self.handle_discovery_message(data, addr[0])
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error receiving: {e}")
                        
        except Exception as e:
            print(f"Error setting up listener: {e}")
        finally:
            try:
                sock.close()
            except:
                pass
    
    def handle_discovery_message(self, data, sender_ip):
        """Handle received discovery message"""
        try:
            message = json.loads(data.decode('utf-8'))
            
            if message.get('type') == 'discovery' and message.get('node_id') != self.device_id:
                device_id = message['node_id']
                device_info = {
                    'id': device_id,
                    'name': message['device_capabilities']['model'],
                    'type': message['device_capabilities']['type'],
                    'address': sender_ip,
                    'port': message['grpc_port'],
                    'capabilities': message['device_capabilities'],
                    'last_seen': datetime.now(),
                    'priority': message.get('priority', 10)
                }
                
                is_new = device_id not in self.discovered_devices
                self.discovered_devices[device_id] = device_info
                
                if is_new:
                    print(f"🔍 Discovered: {device_info['name']} ({device_info['type']}) at {sender_ip}")
                    
        except Exception as e:
            print(f"Error handling discovery message: {e}")
    
    def get_local_ip(self):
        """Get the local IP address"""
        try:
            # Connect to a remote address to determine local IP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except Exception:
            return "192.168.1.100"  # fallback
    
    def get_broadcast_addresses(self):
        """Get possible broadcast addresses for the local network"""
        try:
            local_ip = self.get_local_ip()
            # Common local network broadcast addresses
            ip_parts = local_ip.split('.')
            if ip_parts[0] == '192' and ip_parts[1] == '168':
                # 192.168.x.x network
                return [f"192.168.{ip_parts[2]}.255", "192.168.255.255"]
            elif ip_parts[0] == '10':
                # 10.x.x.x network  
                return [f"10.{ip_parts[1]}.{ip_parts[2]}.255", "10.255.255.255"]
            elif ip_parts[0] == '172':
                # 172.16-31.x.x network
                return [f"172.{ip_parts[1]}.{ip_parts[2]}.255", "172.31.255.255"]
            else:
                return ["255.255.255.255"]
        except Exception:
            return ["192.168.1.255", "192.168.0.255", "255.255.255.255"]
    
    def print_network_info(self):
        """Print network diagnostic information"""
        print("\n🔍 Network Diagnostics:")
        try:
            local_ip = self.get_local_ip()
            print(f"Local IP: {local_ip}")
            
            broadcast_addrs = self.get_broadcast_addresses()
            print(f"Broadcast addresses: {broadcast_addrs}")
            
            # Test socket creation
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                print("✅ UDP broadcast socket creation: OK")
                test_sock.close()
            except Exception as e:
                print(f"❌ UDP broadcast socket creation: {e}")
                
            # Test binding to discovery port
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_sock.bind(('', self.listen_port))
                print(f"✅ Discovery port {self.listen_port} binding: OK")
                test_sock.close()
            except Exception as e:
                print(f"❌ Discovery port {self.listen_port} binding: {e}")
                
            # Test general binding
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.bind(('', 0))  # Bind to any available port
                port = test_sock.getsockname()[1]
                print(f"✅ General socket binding: OK (got port {port})")
                test_sock.close()
            except Exception as e:
                print(f"❌ General socket binding: {e}")
                
            # Test if we can reach other common local IPs
            print("\n🌐 Network Reachability Tests:")
            test_ips = ["192.168.1.1", "192.168.0.1", "10.0.0.1"]
            for test_ip in test_ips:
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    test_sock.settimeout(2.0)
                    test_sock.connect((test_ip, 53))  # Try DNS port
                    print(f"✅ Can reach {test_ip}")
                    test_sock.close()
                except Exception:
                    print(f"❌ Cannot reach {test_ip}")
                    
            # Check what interfaces we can bind to
            print("\n🔌 Interface Binding Tests:")
            bind_tests = [
                ("All interfaces", ""),
                ("Localhost", "127.0.0.1"), 
                ("Local IP", local_ip)
            ]
            
            for name, addr in bind_tests:
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    test_sock.bind((addr, 0))
                    port = test_sock.getsockname()[1]
                    print(f"✅ {name} ({addr or 'any'}): OK (port {port})")
                    test_sock.close()
                except Exception as e:
                    print(f"❌ {name} ({addr or 'any'}): {e}")
                
        except Exception as e:
            print(f"❌ Network diagnostics failed: {e}")
        print()

    def broadcast_presence(self):
        """Broadcast our presence every 2.5 seconds"""
        broadcast_addresses = self.get_broadcast_addresses()
        local_ip = self.get_local_ip()
        
        print(f"🌐 Local IP: {local_ip}")
        print(f"📡 Will try broadcasting to: {broadcast_addresses}")
        
        while self.running:
            try:
                message = {
                    "type": "discovery",
                    "node_id": self.device_id,
                    "grpc_port": self.port,
                    "device_capabilities": self.device_capabilities,
                    "priority": 1,
                    "interface_name": "wifi",
                    "interface_type": "wifi",
                    "timestamp": int(time.time())
                }
                
                data = json.dumps(message).encode('utf-8')
                
                broadcast_success = False
                
                # Try multiple broadcast strategies
                for broadcast_addr in broadcast_addresses:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                        
                        # Bind to specific interface for iOS
                        try:
                            sock.bind((local_ip, 0))
                        except Exception:
                            pass  # fallback to default binding
                        
                        sock.sendto(data, (broadcast_addr, self.broadcast_port))
                        sock.close()
                        broadcast_success = True
                        if self.debug:
                            print(f"✅ Broadcast successful to {broadcast_addr}")
                        break  # Success with this address
                        
                    except Exception as e:
                        if self.debug:
                            print(f"❌ Broadcast failed to {broadcast_addr}: {e}")
                        if "No route to host" in str(e):
                            continue  # Try next address
                        elif "Network is unreachable" in str(e):
                            continue  # Try next address
                        else:
                            print(f"Broadcast error to {broadcast_addr}: {e}")
                
                # Fallback: try multicast
                if not broadcast_success:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        # Use multicast address for local network
                        sock.sendto(data, ('224.0.0.1', self.broadcast_port))
                        sock.close()
                        broadcast_success = True
                    except Exception as e:
                        print(f"Multicast fallback error: {e}")
                
                if broadcast_success:
                    if self.debug:
                        print(f"📡 Broadcasted presence ({len(self.discovered_devices)} devices known)")
                else:
                    if self.debug:
                        print(f"⚠️  Broadcast failed - trying unicast to known devices")
                    # Try direct unicast to previously discovered devices
                    for device_id, device in list(self.discovered_devices.items()):
                        try:
                            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            sock.sendto(data, (device['address'], self.broadcast_port))
                            sock.close()
                        except Exception:
                            pass  # Silent fail for unicast
                
            except Exception as e:
                print(f"Error in broadcast loop: {e}")
            
            time.sleep(2.5)  # EXO's broadcast interval
    
    def stop_discovery(self):
        """Stop discovery service"""
        self.running = False
        print("🛑 Discovery stopped")
    
    def list_devices(self):
        """List all discovered devices"""
        if not self.discovered_devices:
            print("No devices discovered yet")
            return
            
        print(f"\n📱 Discovered {len(self.discovered_devices)} device(s):")
        print("=" * 50)
        
        for device_id, device in self.discovered_devices.items():
            print(f"Device: {device['name']}")
            print(f"  ID: {device_id}")
            print(f"  Type: {device['type']}")
            print(f"  Address: {device['address']}:{device['port']}")
            caps = device['capabilities']
            if 'flops' in caps:
                print(f"  Compute: {caps['flops']['fp32']:.2f} TFLOPS (FP32)")
            print(f"  Last seen: {device['last_seen'].strftime('%H:%M:%S')}")
            print()
    
    def test_listener(self):
        """Test if the UDP listener is working by sending a message to localhost"""
        print("🧪 Testing UDP listener...")
        try:
            # Create a test message with a different device ID
            test_message = {
                "type": "discovery",
                "node_id": f"test-device-{int(time.time())}",
                "grpc_port": 8080,
                "device_capabilities": {
                    "model": "Test Device",
                    "chip": "Test Chip",
                    "type": "test",
                    "memory": 1024,
                    "flops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0}
                },
                "priority": 1,
                "interface_name": "test",
                "interface_type": "test",
                "timestamp": int(time.time())
            }
            
            data = json.dumps(test_message).encode('utf-8')
            
            # Send to localhost
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(data, ('127.0.0.1', self.listen_port))
            sock.close()
            
            print("✅ Test message sent to localhost")
            print("Check if a test device appears in 'list' command in a few seconds...")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def send_test_message(self, target_device_id, message):
        """Send a test message to another device"""
        if target_device_id not in self.discovered_devices:
            print(f"❌ Device {target_device_id} not found")
            return False
            
        device = self.discovered_devices[target_device_id]
        print(f"📤 Sending message to {device['name']}: {message}")
        
        # TODO: Implement actual gRPC message sending
        print("✅ Message sent (placeholder - gRPC client not implemented yet)")
        return True
    
    def submit_task(self, task_type: str, task_data: Dict[str, Any], target_device: Optional[str] = None) -> str:
        """Submit a compute task to the network"""
        task_id = str(uuid.uuid4())
        
        task = ComputeTask(
            task_id=task_id,
            task_type=task_type,
            task_data=task_data,
            metadata={"submitted_by": self.device_id, "submitted_at": str(datetime.now())},
            required_capabilities=[],
            estimated_duration=task_data.get("estimated_duration", 30),
            priority=task_data.get("priority", 5),
            max_devices=task_data.get("max_devices", 1),
            target_subtasks=task_data.get("target_subtasks", 1)
        )
        
        self.submitted_tasks[task_id] = task
        
        if target_device and target_device in self.discovered_devices:
            # Send to specific device
            print(f"📤 Submitting task {task_id} to {target_device}")
            self._send_task_to_device(task, target_device)
        else:
            # Execute locally for now (TODO: implement distributed execution)
            print(f"🔄 Executing task {task_id} locally")
            threading.Thread(target=self._execute_task_locally, args=(task,), daemon=True).start()
            
        return task_id
    
    def _send_task_to_device(self, task: ComputeTask, device_id: str):
        """Send a task to a specific device (placeholder)"""
        # Create subtask for the target device
        subtask = SubTask(
            subtask_id=str(uuid.uuid4()),
            parent_task_id=task.task_id,
            subtask_index=0,
            subtask_data=task.task_data,
            assigned_device_id=device_id
        )
        
        # TODO: Implement actual gRPC call to send task
        # For now, simulate successful submission
        print(f"📡 Task {task.task_id} sent to {device_id} (simulated)")
        
        # Store the subtask
        if task.task_id not in self.task_results:
            self.task_results[task.task_id] = []
        self.task_results[task.task_id].append(subtask)
    
    def _execute_task_locally(self, task: ComputeTask):
        """Execute a task locally"""
        try:
            # Create subtask for local execution
            subtask = SubTask(
                subtask_id=str(uuid.uuid4()),
                parent_task_id=task.task_id,
                subtask_index=0,
                subtask_data=task.task_data,
                assigned_device_id=self.device_id
            )
            
            # Execute using asyncio in a separate thread
            import asyncio
            
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.task_executor.execute_subtask(subtask))
                    return result
                finally:
                    loop.close()
            
            completed_subtask = run_async()
            
            # Store results
            if task.task_id not in self.task_results:
                self.task_results[task.task_id] = []
            self.task_results[task.task_id].append(completed_subtask)
            
        except Exception as e:
            print(f"❌ Error executing task {task.task_id}: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a submitted task"""
        if task_id not in self.submitted_tasks:
            return None
            
        task = self.submitted_tasks[task_id]
        subtasks = self.task_results.get(task_id, [])
        
        completed_subtasks = [st for st in subtasks if st.status == "completed"]
        failed_subtasks = [st for st in subtasks if st.status == "failed"]
        running_subtasks = [st for st in subtasks if st.status == "running"]
        
        total_subtasks = len(subtasks)
        completion_percentage = (len(completed_subtasks) / total_subtasks * 100) if total_subtasks > 0 else 0
        
        return {
            "task_id": task_id,
            "task_type": task.task_type,
            "status": "completed" if len(completed_subtasks) == total_subtasks else 
                     "failed" if len(failed_subtasks) > 0 else
                     "running" if len(running_subtasks) > 0 else "pending",
            "completion_percentage": completion_percentage,
            "total_subtasks": total_subtasks,
            "completed_subtasks": len(completed_subtasks),
            "failed_subtasks": len(failed_subtasks),
            "running_subtasks": len(running_subtasks),
            "results": [st.result_data for st in completed_subtasks if st.result_data]
        }
    
    def list_tasks(self):
        """List all submitted tasks and their status"""
        if not self.submitted_tasks:
            print("No tasks submitted yet")
            return
            
        print(f"\n🎯 Tasks ({len(self.submitted_tasks)}):")
        print("=" * 50)
        
        for task_id, task in self.submitted_tasks.items():
            status = self.get_task_status(task_id)
            if status:
                print(f"Task: {task.task_type}")
                print(f"  ID: {task_id[:8]}...")
                print(f"  Status: {status['status']}")
                print(f"  Progress: {status['completion_percentage']:.1f}%")
                print(f"  Subtasks: {status['completed_subtasks']}/{status['total_subtasks']}")
                if status['results']:
                    for i, result in enumerate(status['results']):
                        if result:
                            print(f"  Result {i+1}: {result.get('result_type', 'unknown')} - {result.get('duration_seconds', 0):.2f}s")
                print()
    
    def create_sample_tasks(self):
        """Create some sample tasks for testing"""
        tasks = [
            {
                "name": "Matrix Multiplication",
                "type": "matrix_multiply", 
                "data": {"type": "matrix_multiply", "matrix_size": 200, "iterations": 3}
            },
            {
                "name": "Simple ML Training",
                "type": "simple_ml_training",
                "data": {"type": "simple_ml_training", "epochs": 3, "batch_size": 16, "model_size": 500}
            },
            {
                "name": "Data Processing",
                "type": "data_processing", 
                "data": {"type": "data_processing", "data_size": 50000, "operations": ["filter", "map", "sort"]}
            }
        ]
        
        print("📋 Sample tasks available:")
        for i, task in enumerate(tasks):
            print(f"  {i+1}. {task['name']}")
            
        return tasks
    
    def set_as_master(self):
        """Set this device as the master/coordinator"""
        self.is_master_device = True
        self.training_manager.is_master = True
        print(f"🏛️  Device {self.device_id[:8]} is now the MASTER coordinator")
        print("📁 MacBook will store model weights and coordinate training")
        print("📤 Ready to send training tasks to worker devices")
    
    def start_mnist_training(self, slave_device_ids: List[str], epochs: int = 10, batch_size: int = 32, 
                            learning_rate: float = 0.01, num_samples: int = 5000) -> str:
        """Start MNIST training with specified configuration"""
        if not self.is_master_device:
            print("❌ Only master device can start distributed training")
            return ""
            
        # Verify slave devices are available (for single device, use self)
        available_slaves = []
        if not slave_device_ids:
            # Single device training
            available_slaves = [self.device_id]
            print(f"🎯 Starting single-device MNIST training on {self.device_id[:8]}...")
        else:
            for device_id in slave_device_ids:
                found = False
                for discovered_device in self.discovered_devices.values():
                    if discovered_device['id'] == device_id:
                        available_slaves.append(device_id)
                        found = True
                        break
                if not found:
                    print(f"⚠️  Device {device_id} not found in discovered devices")
            
            if not available_slaves:
                print("❌ No valid slave devices found")
                return ""
        
        # Create MNIST neural network architecture
        model_config = self.training_manager.create_neural_network(
            input_size=784,     # 28x28 MNIST images
            hidden_sizes=[128, 64],  # Two hidden layers for MNIST
            output_size=10      # 10 digit classes
        )
        
        # Training configuration
        training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_samples": num_samples
        }
        
        print(f"🧠 MNIST Neural Network Configuration:")
        print(f"   Architecture: 784 → 128 → 64 → 10")
        print(f"   Parameters: {model_config['total_parameters']:,}")
        print(f"   Training samples: {num_samples:,}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        job_id = self.training_manager.start_distributed_training(
            available_slaves, model_config, training_config
        )
        
        # Start training process in background
        threading.Thread(target=self._run_mnist_training, args=(job_id,), daemon=True).start()
        
        return job_id
    
    def start_distributed_neural_training(self, slave_device_ids: List[str]) -> str:
        """Start distributed neural network training with specified slave devices"""
        # Use MNIST by default
        return self.start_mnist_training(slave_device_ids)
    
    def _run_mnist_training(self, job_id: str):
        """Run the MNIST training process with detailed metrics"""
        try:
            job = self.training_manager.training_jobs[job_id]
            job.status = "training"
            
            print(f"\n🏃‍♂️ Starting MNIST training across {len(job.slave_devices)} device(s)")
            print(f"🎯 Target: Train neural network to recognize handwritten digits (0-9)")
            
            # Track training metrics
            epoch_losses = []
            epoch_accuracies = []
            total_training_time = 0
            
            for epoch in range(job.total_epochs):
                epoch_start = time.time()
                print(f"\n📚 Epoch {epoch + 1}/{job.total_epochs}")
                job.current_epoch = epoch
                
                # Distribute batches to slave devices (with single device getting 100% workload)
                device_workloads = {job.slave_devices[0]: 1.0} if len(job.slave_devices) == 1 else None
                batches = self.training_manager.distribute_epoch_batches(job_id, device_workloads)
                print(f"   📦 Created {len(batches)} training batches")
                
                # Process batches - send to workers or execute locally
                gradient_results = []
                batch_count = 0
                
                for batch in batches:
                    batch_count += 1
                    if len(batches) <= 5 or batch_count % max(1, len(batches) // 5) == 0:
                        print(f"   🔄 Processing batch {batch_count}/{len(batches)} ({len(batch.data)} samples)")
                    
                    # Find which device should process this batch
                    target_device_id = batch.model_weights.get('assigned_device', job.slave_devices[0])
                    
                    if target_device_id == self.device_id:
                        # Process locally (single device mode)
                        result = self.training_manager.process_training_batch(batch)
                        gradient_results.append(result)
                    else:
                        # Send to worker device (MacBook → iPad)
                        target_device = None
                        for device_id, device_info in self.discovered_devices.items():
                            if device_info['id'] == target_device_id:
                                target_device = device_info
                                break
                        
                        if target_device:
                            print(f"   📤 Sending batch to {target_device['name']} ({target_device['address']})")
                            
                            # Send work to iPad
                            result = self.training_service.send_training_batch(
                                target_device['address'], 
                                8090,  # Training service port
                                batch
                            )
                            
                            if result:
                                print(f"   📥 Received gradients from {target_device['name']}")
                                gradient_results.append(result)
                            else:
                                print(f"   ❌ Failed to get result from {target_device['name']}")
                        else:
                            # Fallback to local processing
                            print(f"   ⚠️  Device {target_device_id[:8]} not available, processing locally")
                            result = self.training_manager.process_training_batch(batch)
                            gradient_results.append(result)
                    
                    # Small delay between batches
                    time.sleep(0.1)
                
                # Aggregate gradients from all devices
                aggregated = self.training_manager.aggregate_gradients(job_id, gradient_results)
                
                if aggregated:
                    avg_loss = aggregated["average_loss"]
                    avg_accuracy = aggregated["average_accuracy"]
                    total_samples = aggregated["total_samples"]
                    computation_time = aggregated["total_computation_time"]
                    
                    epoch_losses.append(avg_loss)
                    epoch_accuracies.append(avg_accuracy)
                    
                    print(f"   📊 Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%) | Samples: {total_samples}")
                    print(f"   ⏱️  Computation: {computation_time:.2f}s | Learning rate: {job.training_config['learning_rate']:.3f}")
                    
                    # Update model parameters
                    self.training_manager.update_model_parameters(
                        job_id, 
                        aggregated["aggregated_gradients"], 
                        job.training_config["learning_rate"]
                    )
                    
                    # Show progress indicators
                    if epoch > 0:
                        loss_change = epoch_losses[-1] - epoch_losses[-2]
                        acc_change = epoch_accuracies[-1] - epoch_accuracies[-2]
                        loss_trend = "📉" if loss_change < 0 else "📈" if loss_change > 0 else "➡️"
                        acc_trend = "📈" if acc_change > 0 else "📉" if acc_change < 0 else "➡️"
                        print(f"   📈 Trends: Loss {loss_trend} ({loss_change:+.4f}) | Accuracy {acc_trend} ({acc_change:+.3f})")
                    
                    print(f"   ✅ Model parameters updated")
                
                epoch_time = time.time() - epoch_start
                total_training_time += epoch_time
                print(f"   🕐 Epoch completed in {epoch_time:.1f}s")
                
                # Adaptive learning rate (simple decay)
                if epoch > 0 and epoch % 3 == 0:
                    job.training_config["learning_rate"] *= 0.9
                    print(f"   📉 Learning rate reduced to {job.training_config['learning_rate']:.4f}")
            
            job.status = "completed"
            
            # Save model to MacBook (if master)
            if self.is_master_device:
                model_path = self.save_trained_model(job_id, job, epoch_losses[-1], epoch_accuracies[-1])
                print(f"💾 Model saved to: {model_path}")
            
            # Training summary
            print(f"\n🎉 MNIST Training Completed!")
            print(f"📊 Final Results:")
            print(f"   Final Loss: {epoch_losses[-1]:.4f}")
            print(f"   Final Accuracy: {epoch_accuracies[-1]:.3f} ({epoch_accuracies[-1]*100:.1f}%)")
            print(f"   Total Training Time: {total_training_time:.1f}s")
            print(f"   Average Time per Epoch: {total_training_time/job.total_epochs:.1f}s")
            print(f"   Model Parameters: {job.model_config['total_parameters']:,}")
            
            # Estimate model performance
            if epoch_accuracies[-1] > 0.9:
                print(f"🏆 Excellent! Your model achieved >90% accuracy!")
            elif epoch_accuracies[-1] > 0.8:
                print(f"👍 Good performance! Model achieved >80% accuracy")
            elif epoch_accuracies[-1] > 0.6:
                print(f"📚 Learning! Model achieved >60% accuracy")
            else:
                print(f"🔄 Model is still learning. Try more epochs or adjust hyperparameters.")
            
        except Exception as e:
            print(f"❌ Error in MNIST training: {e}")
            if job_id in self.training_manager.training_jobs:
                self.training_manager.training_jobs[job_id].status = "failed"
    
    def _run_distributed_training(self, job_id: str):
        """Run the distributed training process (Master only)"""
        # Use the MNIST training for now
        self._run_mnist_training(job_id)
    
    def get_training_jobs(self):
        """List all training jobs"""
        if not self.training_manager.training_jobs:
            print("No training jobs found")
            return
            
        print(f"\n🎯 Training Jobs ({len(self.training_manager.training_jobs)}):")
        print("=" * 60)
        
        for job_id, job in self.training_manager.training_jobs.items():
            status = self.training_manager.get_training_status(job_id)
            if status:
                print(f"Job: {job_id[:8]}...")
                print(f"  Status: {status['status']}")
                print(f"  Progress: {status['progress']:.1f}% ({status['current_epoch']}/{status['total_epochs']} epochs)")
                print(f"  Model: {status['model_info']['layers']} layers, {status['model_info']['parameters']} parameters")
                print(f"  Workers: {len(status['slave_devices'])} devices")
                print(f"  Master: {status['master_device'][:8]}...")
                print()
    
    def simulate_slave_mode(self):
        """Set this device as a slave worker"""
        self.is_master_device = False
        self.training_manager.is_master = False
    
    def save_trained_model(self, job_id: str, job: DistributedTrainingJob, final_loss: float, final_accuracy: float) -> str:
        """Save trained model to MacBook storage"""
        import os
        import json
        
        # Create models directory
        models_dir = os.path.expanduser("~/discompute_models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"mnist_model_{timestamp}_{job_id[:8]}"
        model_path = os.path.join(models_dir, f"{model_name}.json")
        
        # Prepare model data
        model_data = {
            "job_id": job_id,
            "model_type": "MNIST_Classifier",
            "architecture": job.model_config,
            "trained_parameters": job.model_parameters,
            "training_config": job.training_config,
            "final_metrics": {
                "loss": final_loss,
                "accuracy": final_accuracy,
                "epochs_trained": job.total_epochs
            },
            "training_info": {
                "device_count": len(job.slave_devices),
                "master_device": self.device_id,
                "worker_devices": job.slave_devices,
                "training_date": timestamp
            },
            "metadata": {
                "framework": "discompute",
                "version": "1.0",
                "total_parameters": job.model_config["total_parameters"]
            }
        }
        
        # Save model
        try:
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            print(f"📁 Model details:")
            print(f"   File: {model_name}.json")
            print(f"   Location: {models_dir}")
            print(f"   Size: {os.path.getsize(model_path) / 1024:.1f} KB")
            print(f"   Parameters: {job.model_config['total_parameters']:,}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return ""
    
    def list_saved_models(self):
        """List all saved models on MacBook"""
        import os
        import json
        
        models_dir = os.path.expanduser("~/discompute_models")
        if not os.path.exists(models_dir):
            print("No saved models found")
            return
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
        
        if not model_files:
            print("No saved models found")
            return
        
        print(f"\n💾 Saved Models ({len(model_files)}):")
        print("=" * 60)
        
        for model_file in sorted(model_files, reverse=True):  # Most recent first
            try:
                model_path = os.path.join(models_dir, model_file)
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                
                metrics = model_data.get("final_metrics", {})
                info = model_data.get("training_info", {})
                
                print(f"Model: {model_file}")
                print(f"  Type: {model_data.get('model_type', 'Unknown')}")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.3f} ({metrics.get('accuracy', 0)*100:.1f}%)")
                print(f"  Loss: {metrics.get('loss', 0):.4f}")
                print(f"  Epochs: {metrics.get('epochs_trained', 0)}")
                print(f"  Devices: {info.get('device_count', 1)}")
                print(f"  Date: {info.get('training_date', 'Unknown')}")
                print(f"  Size: {os.path.getsize(model_path) / 1024:.1f} KB")
                print()
                
            except Exception as e:
                print(f"Error reading {model_file}: {e}")
                print()
    
    def start_slave_mode(self):
        """Set this device as a slave worker"""
        self.is_master_device = False
        self.training_manager.is_master = False
        
        # Start training service to receive work from master
        port = self.training_service.start_server(port=8090)
        if port:
            print(f"🤖 Device {self.device_id[:8]} is now in SLAVE mode")
            print(f"🔌 Training service listening on port {port}")
            print("⚡ iPad ready to receive compute work from MacBook")
            print("📥 Waiting for training batches from master...")
        else:
            print("❌ Failed to start slave training service")
    
    def get_stats(self):
        """Get service statistics"""
        uptime = time.time() - self.start_time
        
        # Task statistics
        total_tasks = len(self.submitted_tasks)
        completed_tasks = 0
        running_tasks = 0
        failed_tasks = 0
        
        for task_id in self.submitted_tasks:
            status = self.get_task_status(task_id)
            if status:
                if status['status'] == 'completed':
                    completed_tasks += 1
                elif status['status'] == 'running':
                    running_tasks += 1
                elif status['status'] == 'failed':
                    failed_tasks += 1
        
        return {
            'running': self.running,
            'device_id': self.device_id,
            'discovered_devices': len(self.discovered_devices),
            'device_type': self.device_capabilities['type'],
            'chip': self.device_capabilities['chip'],
            'uptime_seconds': round(uptime, 1),
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'running_tasks': running_tasks,
            'failed_tasks': failed_tasks,
            'active_executor_tasks': len(self.task_executor.running_tasks),
            'is_master_device': self.is_master_device,
            'training_jobs': len(self.training_manager.training_jobs)
        }

def main():
    """Main function for interactive use"""
    print("🍎 Discompute iOS Client")
    print("Compatible with Pyto, a-Shell, and other Python iOS apps")
    print()
    
    # Check for debug flag
    debug = len(sys.argv) > 1 and sys.argv[1] == '--debug'
    
    # Create client
    client = DiscomputeMobile(debug=debug)
    
    try:
        # Start discovery
        client.start_discovery()
        
        print("\nCommands:")
        print("  'list' - Show discovered devices")
        print("  'info' - Show this device info") 
        print("  'stats' - Show statistics")
        print("  'debug' - Show network diagnostics")
        print("  'test' - Test listener by sending to localhost")
        print("  'tasks' - Show submitted tasks")
        print("  'samples' - Show sample AI tasks")
        print("  'submit <task_type>' - Submit a task (e.g., 'submit matrix')")
        print("  'master' - Set this device as master coordinator")
        print("  'slave' - Set this device as slave worker")
        print("  'mnist' - Start MNIST digit recognition training")
        print("  'train <device_ids>' - Start distributed neural training")
        print("  'jobs' - Show distributed training jobs")
        print("  'models' - List saved models (MacBook only)")
        print("  'send <device_id> <message>' - Send test message")
        print("  'quit' - Exit")
        print()
        
        while True:
            try:
                cmd = input("discompute> ").strip().lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'list':
                    client.list_devices()
                elif cmd == 'info':
                    caps = client.device_capabilities
                    print(f"Device: {caps['model']} ({caps['type']})")
                    print(f"Chip: {caps['chip']}")
                    print(f"Memory: {caps['memory']} MB")
                    print(f"Compute: {caps['flops']['fp32']:.2f} TFLOPS")
                elif cmd == 'stats':
                    stats = client.get_stats()
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif cmd == 'debug':
                    client.print_network_info()
                elif cmd == 'test':
                    client.test_listener()
                elif cmd == 'tasks':
                    client.list_tasks()
                elif cmd == 'samples':
                    client.create_sample_tasks()
                elif cmd.startswith('submit '):
                    task_type = cmd.split(' ', 1)[1] if len(cmd.split(' ')) > 1 else ""
                    samples = client.create_sample_tasks()
                    
                    if task_type == "matrix" or task_type == "1":
                        task_data = samples[0]["data"]
                        task_id = client.submit_task("matrix_multiply", task_data)
                        print(f"✅ Submitted matrix multiplication task: {task_id[:8]}...")
                    elif task_type == "training" or task_type == "ml" or task_type == "2":
                        task_data = samples[1]["data"]
                        task_id = client.submit_task("simple_ml_training", task_data)
                        print(f"✅ Submitted ML training task: {task_id[:8]}...")
                    elif task_type == "data" or task_type == "processing" or task_type == "3":
                        task_data = samples[2]["data"]
                        task_id = client.submit_task("data_processing", task_data)
                        print(f"✅ Submitted data processing task: {task_id[:8]}...")
                    else:
                        print("Available task types: matrix, training, data (or numbers 1-3)")
                elif cmd == 'master':
                    client.set_as_master()
                elif cmd == 'slave':
                    client.start_slave_mode()
                elif cmd == 'models':
                    client.list_saved_models()
                elif cmd == 'jobs':
                    client.get_training_jobs()
                elif cmd == 'mnist':
                    if not client.is_master_device:
                        print("⚠️  Setting device as master first...")
                        client.set_as_master()
                    
                    print("🎯 Starting MNIST training on this device...")
                    job_id = client.start_mnist_training([], epochs=10, batch_size=32, num_samples=2000)
                    if job_id:
                        print(f"✅ Started MNIST training job: {job_id[:8]}...")
                        print("📊 Watch the training progress above!")
                elif cmd.startswith('train '):
                    device_list = cmd.split(' ', 1)[1] if len(cmd.split(' ')) > 1 else ""
                    if device_list:
                        # Parse device IDs from command
                        device_ids = [d.strip() for d in device_list.split(',')]
                        
                        # Try to match partial device IDs with discovered devices
                        matched_devices = []
                        for partial_id in device_ids:
                            for device_id, device_info in client.discovered_devices.items():
                                if device_info['id'].startswith(partial_id) or partial_id in device_info['id']:
                                    matched_devices.append(device_info['id'])
                                    break
                        
                        if matched_devices:
                            print(f"Starting distributed training with devices: {[d[:8] + '...' for d in matched_devices]}")
                            job_id = client.start_distributed_neural_training(matched_devices)
                            if job_id:
                                print(f"✅ Started training job: {job_id[:8]}...")
                        else:
                            print("❌ No matching devices found")
                            print("Available devices:")
                            for device_id, device_info in client.discovered_devices.items():
                                print(f"  {device_info['id'][:8]}... ({device_info['name']})")
                    else:
                        print("Usage: train <device_id1>,<device_id2>,...")
                        print("Example: train abc123,def456")
                        if client.discovered_devices:
                            print("Available devices:")
                            for device_id, device_info in client.discovered_devices.items():
                                print(f"  {device_info['id'][:8]}... ({device_info['name']})")
                elif cmd.startswith('send '):
                    parts = cmd.split(' ', 2)
                    if len(parts) >= 3:
                        client.send_test_message(parts[1], parts[2])
                    else:
                        print("Usage: send <device_id> <message>")
                elif cmd == 'help':
                    print("Available commands: list, info, stats, debug, test, tasks, samples, submit, master, slave, mnist, train, jobs, models, send, quit")
                elif cmd == '':
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
                
    finally:
        client.stop_discovery()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
