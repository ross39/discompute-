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

# Check for iOS client
IOS_CLIENT="ios-client/discompute_ios_enhanced.py"
if [ ! -f "$IOS_CLIENT" ]; then
    echo "❌ iOS client not found: $IOS_CLIENT"
    exit 1
fi

echo "📱 iOS client found: $IOS_CLIENT"

echo ""
echo "📋 Setup Instructions:"
echo "======================"
echo ""
echo "1. 🍎 iOS Device Setup:"
echo "   • Install Pythonista 3 from the App Store (recommended)"
echo "   • Or install a-Shell or Pyto for Python support"
echo "   • Copy ios-client/discompute_ios_enhanced.py to your iOS device"
echo "   • Install required packages:"
echo "     - pip install tinygrad numpy psutil"
echo ""
echo "2. 🌐 Network Setup:"
echo "   • Ensure all devices are on the same Wi-Fi network"
echo "   • Make sure UDP port 5005 is not blocked"
echo "   • iOS devices will broadcast on this port"
echo ""
echo "3. 🚀 Running the Demo:"
echo "   • First: Start the iOS client on each device"
echo "   • Then: Run this training demo on your computer"
echo ""

read -p "📱 Have you started the iOS clients? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔧 To start the iOS client:"
    echo "   1. Open Pythonista on your iOS device"
    echo "   2. Copy and run: discompute_ios_enhanced.py"
    echo "   3. The client will start broadcasting and wait for training jobs"
    echo ""
    echo "📱 Example output on iOS:"
    echo "   🚀 Starting Enhanced Discompute iOS Client"
    echo "   ✅ iOS client running on device: iPad Pro"
    echo "   📡 Broadcasting on UDP port 5005"
    echo "   🔗 HTTP training server on port 8080"
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
echo "• Discovered iOS devices on the network"
echo "• Distributed MNIST training across devices"
echo "• Each device trained on different data batches"
echo "• Gradients were aggregated for model updates"
echo "• Training completed with distributed neural network!"
echo ""
echo "🔧 Troubleshooting:"
echo "• No devices found? Check Wi-Fi connection and iOS client status"
echo "• Training failed? Check iOS device logs and battery levels"
echo "• Network issues? Ensure UDP port 5005 is open"
echo ""
echo "📚 Next Steps:"
echo "• Scale to more iOS devices for faster training"
echo "• Try different models and datasets"
echo "• Implement model parallel training for larger models"
echo "• Add real-time monitoring and visualization"
