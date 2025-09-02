#!/bin/bash

# Discompute Distributed MNIST Training Demo
# This script demonstrates distributed neural network training across iOS devices

set -e

echo "🚀 Discompute Distributed MNIST Training Demo"
echo "=============================================="

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.19+ first."
    exit 1
fi

# Build the discompute binary
echo "🔨 Building discompute binary..."
cd "$(dirname "$0")/.."
go build -o bin/discompute cmd/discompute/main.go

if [ ! -f "bin/discompute" ]; then
    echo "❌ Failed to build discompute binary"
    exit 1
fi

echo "✅ Discompute binary built successfully"

# Check for Mac worker
MAC_WORKER="mac-worker/discompute_mac_worker.py"
if [ ! -f "$MAC_WORKER" ]; then
    echo "❌ Mac worker not found: $MAC_WORKER"
    exit 1
fi

echo "💻 Mac worker found: $MAC_WORKER"

echo ""
echo "📋 Setup Instructions:"
echo "======================"
echo ""
echo "1. 🍎 Mac Device Setup:"
echo "   • Install Python 3.8+ on each Mac"
echo "   • Install ML frameworks (MLX recommended for Apple Silicon):"
echo "     - pip install mlx tinygrad torch psutil numpy"
echo "   • Copy mac-worker/discompute_mac_worker.py to each Mac"
echo ""
echo "2. 🌐 Network Setup:"
echo "   • Ensure all Macs are on the same Wi-Fi network"
echo "   • Make sure UDP port 5005 is not blocked"
echo "   • Mac devices will broadcast on this port"
echo ""
echo "3. 🚀 Running the Demo:"
echo "   • First: Start the Mac worker on each device"
echo "   • Then: Run this training demo on your coordinator Mac"
echo ""

read -p "💻 Have you started the Mac workers? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔧 To start the Mac worker:"
    echo "   1. Open Terminal on each Mac"
    echo "   2. Run: python mac-worker/discompute_mac_worker.py"
    echo "   3. The worker will start broadcasting and wait for training jobs"
    echo ""
    echo "💻 Example output on Mac:"
    echo "   🚀 Discompute Mac Worker Client"
    echo "   ✅ Mac worker running"
    echo "   📊 Device: MacBook Pro (Apple M3 Max)"
    echo "   📡 Broadcasting on UDP port 5005"
    echo "   🔗 HTTP server on port 8080"
    echo "   💡 Ready for distributed training!"
    echo ""
    exit 0
fi

echo ""
echo "🔍 Starting device discovery and training..."
echo ""

# Start the distributed training
./bin/discompute \
    -mode=training \
    -training-model=mnist_cnn \
    -training-epochs=5 \
    -training-batch=32 \
    -training-lr=0.001 \
    -max-devices=4 \
    -log-level=info

echo ""
echo "🎉 Demo completed!"
echo ""
echo "📊 What happened:"
echo "• Discovered Mac devices on the network"
echo "• Distributed MNIST training across Macs"
echo "• Each Mac trained on different data batches"
echo "• Gradients were aggregated for model updates"
echo "• Training completed with distributed neural network!"
echo ""
echo "🔧 Troubleshooting:"
echo "• No devices found? Check Wi-Fi connection and Mac worker status"
echo "• Training failed? Check Mac device logs and available memory"
echo "• Network issues? Ensure UDP port 5005 is open"
echo ""
echo "📚 Next Steps:"
echo "• Scale to more Mac devices for faster training"
echo "• Try different models and datasets"
echo "• Implement model parallel training for larger models"
echo "• Add real-time monitoring and visualization"
echo "• Optimize for specific Apple Silicon chips (M1/M2/M3)"
