#!/bin/bash

# Build script for iOS deployment
# This creates a cross-compiled binary that can run on iOS

echo "Building discompute for iOS..."

# Set iOS build environment
export GOOS=ios
export GOARCH=arm64
export CGO_ENABLED=1

# Build the binary
go build -o bin/discompute-ios ./cmd/discompute

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ iOS binary built successfully: bin/discompute-ios"
    echo ""
    echo "To deploy to iPhone/iPad:"
    echo "1. Use a tool like 'ios-deploy' or Xcode"
    echo "2. Copy the binary to your iOS device"
    echo "3. Run with: ./discompute-ios start --port 8080"
    echo ""
    echo "Note: This requires a jailbroken device or enterprise certificate"
    echo "For App Store distribution, you'll need to create a native iOS app wrapper"
else
    echo "❌ iOS build failed"
    exit 1
fi
