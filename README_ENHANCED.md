# Discompute: Distributed Neural Network Training on iOS Devices

## Overview

Discompute enables distributed neural network training across iOS devices using **tinygrad** for high-performance computation and **Go** for efficient networking and orchestration. Turn your iPhone and iPad devices into a powerful compute cluster!

## 🚀 Key Features

- **iOS Device Support**: Run neural network training on iPhones and iPads
- **Tinygrad Integration**: High-performance ML computation optimized for mobile
- **Automatic Discovery**: Devices automatically find each other via UDP broadcasting
- **Distributed Training**: Data-parallel training across multiple devices
- **Real-time Monitoring**: Track training progress and device metrics
- **Battery-Aware**: Respects device battery levels and charging status
- **Go + Python Hybrid**: Go for networking, Python for ML workloads

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Discompute Master                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   UDP Discovery │  │  gRPC Server    │  │  Trainer    │ │
│  │   (Go)          │  │  (Go)           │  │  (Go)       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
         ┌─────────────────┐ │ ┌─────────────────┐
         │   iOS Device 1  │ │ │   iOS Device N  │
         │                 │ │ │                 │
         │ ┌─────────────┐ │ │ │ ┌─────────────┐ │
         │ │HTTP Server  │ │ │ │ │HTTP Server  │ │
         │ │(Python)     │ │ │ │ │(Python)     │ │
         │ └─────────────┘ │ │ │ └─────────────┘ │
         │ ┌─────────────┐ │ │ │ ┌─────────────┐ │
         │ │Tinygrad ML  │ │ │ │ │Tinygrad ML  │ │
         │ │Engine       │ │ │ │ │Engine       │ │
         │ └─────────────┘ │ │ │ └─────────────┘ │
         └─────────────────┘ │ └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- **Go 1.19+** (for the master node)
- **iOS devices** with Python support:
  - [Pythonista 3](https://apps.apple.com/app/pythonista-3/id1085978097) (recommended)
  - [a-Shell](https://apps.apple.com/app/a-shell/id1473805438) with Python
  - [Pyto](https://apps.apple.com/app/pyto/id1436650069)

### Master Node Setup (Mac/Linux/Windows)

```bash
# Clone the repository
git clone https://github.com/yourusername/discompute
cd discompute

# Build the Go binary
go build -o bin/discompute cmd/discompute/main.go

# Make demo script executable
chmod +x scripts/demo_mnist_training.sh
```

### iOS Device Setup

1. **Install a Python environment on iOS:**
   - **Pythonista 3** (recommended): Full-featured Python IDE
   - **a-Shell**: Terminal with Python support
   - **Pyto**: Free Python IDE

2. **Install required packages:**
   ```python
   # In your iOS Python environment
   import subprocess
   subprocess.run(['pip', 'install', 'tinygrad', 'numpy', 'psutil'])
   ```

3. **Copy the iOS client:**
   - Transfer `ios-client/discompute_ios_enhanced.py` to your iOS device
   - You can use AirDrop, email, or cloud storage

## 🚀 Quick Start

### 1. Start iOS Clients

On each iOS device:

```python
# Run in Pythonista, a-Shell, or Pyto
python discompute_ios_enhanced.py
```

Expected output:
```
🚀 Starting Enhanced Discompute iOS Client
==================================================
🧠 Tinygrad available - neural network training enabled
📱 iOS-specific features available
✅ iOS client running on device: iPad Pro
📡 Broadcasting on UDP port 5005
🔗 HTTP training server on port 8080
🆔 Device ID: ios_a1b2c3d4
💡 Ready for distributed training!
```

### 2. Start Distributed Training

On your master node (Mac/Linux/Windows):

```bash
# Run the demo
./scripts/demo_mnist_training.sh
```

Or manually:
```bash
./bin/discompute \
    -mode=training \
    -training-model=mnist_cnn \
    -training-epochs=10 \
    -training-batch=32 \
    -max-devices=4
```

### 3. Watch the Magic! ✨

The system will:
1. Discover iOS devices on your network
2. Initialize neural network training on each device
3. Distribute MNIST training batches across devices
4. Aggregate gradients and update models
5. Complete distributed training!

## 📱 Supported iOS Environments

| Environment | Status | Notes |
|-------------|--------|-------|
| **Pythonista 3** | ✅ Recommended | Full iOS API access, best performance |
| **a-Shell** | ✅ Supported | Terminal-based, good for automation |
| **Pyto** | ✅ Supported | Free alternative, basic features |
| **Juno** | ⚠️ Experimental | Jupyter notebooks on iOS |

## 🧠 Neural Network Models

Currently supported models:

### MNIST CNN
- **Architecture**: Conv2D → Conv2D → Dense → Dense
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Performance**: ~98% accuracy in 5-10 epochs

### Custom Models
Add your own models by:
1. Implementing the model in `TinygradMNISTModel` class
2. Adding configuration in `TrainingConfig`
3. Update the model factory in the iOS client

## 🔧 Configuration

### Training Parameters

```bash
./bin/discompute -mode=training \
    -training-model=mnist_cnn \     # Model type
    -training-epochs=10 \           # Number of epochs
    -training-batch=32 \            # Batch size per device
    -training-lr=0.001 \            # Learning rate
    -max-devices=4 \                # Max iOS devices to use
    -log-level=info                 # Logging level
```

### Network Settings

```bash
./bin/discompute \
    -listen-port=5005 \             # UDP discovery port
    -broadcast-port=5005 \          # UDP broadcast port
    -node-port=50051 \              # gRPC server port
    -http-port=8080                 # HTTP API port
```

## 📊 Monitoring and Metrics

### Real-time Training Metrics
- **Loss**: Cross-entropy loss per epoch
- **Accuracy**: Classification accuracy
- **Device Metrics**: CPU, memory, battery usage
- **Throughput**: Tensors processed per second

### Device Health Monitoring
- Battery level tracking
- Temperature monitoring
- Memory usage alerts
- Automatic device removal on low battery

## 🛠️ Advanced Usage

### Custom Training Data

```python
# On iOS client - modify loadTrainingData function
def load_custom_data():
    # Load your custom dataset
    X_train = load_your_data()
    y_train = load_your_labels()
    return X_train, y_train
```

### Model Parallel Training

```go
// In Go server - implement model sharding
config := training.TrainingConfig{
    DistributionMode: "model_parallel",
    // Split model layers across devices
}
```

### Multi-Network Training

```bash
# Train across multiple Wi-Fi networks
./bin/discompute -mode=training -discovery-timeout=60
```

## 🔍 Troubleshooting

### Common Issues

**No iOS devices discovered:**
- Check Wi-Fi connection (all devices on same network)
- Verify UDP port 5005 is not blocked
- Restart iOS Python client
- Check iOS device logs

**Training fails:**
- Ensure iOS devices have >20% battery
- Check tinygrad installation: `import tinygrad`
- Verify numpy installation: `import numpy`
- Monitor iOS device memory usage

**Performance issues:**
- Reduce batch size for older devices
- Close other iOS apps to free memory
- Use fewer devices if network is slow
- Enable airplane mode + Wi-Fi for better performance

### Debug Mode

```bash
./bin/discompute -mode=training -log-level=debug
```

### iOS Client Logs

```python
# In iOS Python environment
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **New Models**: Implement more neural network architectures
- **iOS Optimization**: Better performance on older devices  
- **Network Improvements**: Fault tolerance and recovery
- **Visualization**: Real-time training dashboards
- **Testing**: More comprehensive test coverage

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/discompute
cd discompute

# Install Go dependencies
go mod tidy

# Run tests
go test ./...

# Build and test iOS client
cd ios-client
python test_tinygrad_mnist.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Tinygrad](https://github.com/tinygrad/tinygrad)**: High-performance ML framework
- **[EXO](https://github.com/exo-explore/exo)**: Inspiration for distributed AI
- **iOS Python Communities**: Pythonista, a-Shell, and Pyto developers
- **Go Community**: Excellent networking and concurrency libraries

## 🚀 What's Next?

- **Apple Silicon Optimization**: Native Metal compute shaders
- **Model Hub**: Pre-trained model sharing
- **Visual Training**: Real-time loss/accuracy graphs
- **Cloud Integration**: Hybrid cloud + edge training
- **More Platforms**: Android and desktop support

---

**Ready to turn your iOS devices into a neural network training cluster?** 

Start with the quick start guide above! 🚀📱🧠
