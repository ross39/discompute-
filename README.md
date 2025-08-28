# Discompute - Distributed Compute System

A distributed compute system that enables AI/ML workloads to be distributed across multiple devices including MacBooks, iPhones, iPads, and other platforms. Built with Go and inspired by the EXO project architecture.

## Project Status

**✅ Milestone 1 COMPLETED: Device Discovery and Communication**

**Completed Features:**
- ✅ **UDP-based Device Discovery** (following EXO's proven pattern)
- ✅ **Apple Device Capability Detection** (M1-M4, A13-A18 Pro support)
- ✅ **Cross-platform gRPC Communication** 
- ✅ **Peer-to-peer Architecture** (no central coordinator required)
- ✅ **iOS/iPad Binary Support** (direct iOS deployment)
- ✅ **Advanced Device Registry** with compute metrics
- ✅ **CLI Tools** for service management and testing
- ✅ **Mobile Framework** (gomobile compatible)

**Ready for Testing:**
- Mac ↔ iPhone 16 Pro communication
- Mac ↔ iPad Pro communication  
- Cross-platform device discovery
- Compute capability assessment

📋 **Next Steps (Milestone 2):**
- Task scheduling and load balancing
- Compute task distribution framework
- Background processing optimization

## Features

### Milestone 1: Device Discovery and Communication
- **mDNS Discovery**: Automatic discovery of devices on the local network
- **gRPC Communication**: Secure, efficient communication between devices
- **Device Registry**: Central registry of available compute devices
- **Cross-Platform**: Support for macOS, Linux, Windows, iOS, and Android
- **CLI Interface**: Command-line tools for management and testing

## Quick Start

### Prerequisites

- Go 1.21 or later
- Protocol Buffers compiler (`protoc`)
- Required Go packages for protobuf generation

### Setup

1. **Install required tools:**
   ```bash
   # Install protoc (macOS)
   brew install protobuf
   
   # Install Go protobuf generators
   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
   ```

2. **Clone and build:**
   ```bash
   git clone https://github.com/rossheaney/discompute
   cd discompute
   make setup
   make build
   ```

### Usage

#### Mac/Linux/Windows Usage

Start the discompute service on your desktop/server:

```bash
# Start with default settings
./bin/discompute start

# Start with custom port and debug logging  
./bin/discompute start --port 9090 --debug

# Show device capabilities
./bin/discompute info
```

#### iPhone/iPad Usage  

Deploy and run on iOS devices:

```bash
# Build for iOS
./scripts/build-ios.sh

# Copy bin/discompute-ios to your iPhone/iPad
# Via Xcode, ios-deploy, or file transfer

# On iPhone/iPad, run:
./discompute-ios start --port 8080 --debug
```

**iOS Deployment Options:**
- **Jailbroken devices**: Direct binary execution
- **Enterprise certificate**: Deploy via enterprise MDM
- **TestFlight**: Package as iOS app for testing
- **Development**: Use Xcode for development builds

#### List Discovered Devices

See what devices are available on your network:

```bash
./bin/discompute list
```

Example output:
```
Found 2 device(s):

Device ID: a1b2c3d4e5f6g7h8
  Name: MacBook Pro
  Type: mac
  Address: 192.168.1.100:8080
  Status: AVAILABLE
  Capabilities:
    CPU Cores: 12
    Memory: 65536 MB
    GPU: true (Apple M2 Max)
    Compute Performance:
      FP32: 13.49 TFLOPS
      FP16: 26.98 TFLOPS
      INT8: 53.96 TFLOPS
  Last Seen: 2024-08-28T18:30:45Z

Device ID: i9j8k7l6m5n4o3p2
  Name: iPhone16,2
  Type: iphone
  Address: 192.168.1.101:8080
  Status: AVAILABLE
  Capabilities:
    CPU Cores: 6
    Memory: 8192 MB
    GPU: true (Apple A18 Pro)
    Compute Performance:
      FP32: 2.80 TFLOPS
      FP16: 5.60 TFLOPS
      INT8: 11.20 TFLOPS
    Battery: 85.0%
  Last Seen: 2024-08-28T18:30:42Z
```

#### Send Test Messages

Send a test message between devices:

```bash
# Send a message to another device
./bin/discompute send i9j8k7l6m5n4o3p2 "Hello from Mac!"
```

## Architecture

### Project Structure

```
discompute/
├── cmd/discompute/          # CLI application
├── internal/
│   ├── client/              # gRPC client implementation
│   ├── server/              # gRPC server implementation
│   ├── discovery/           # mDNS device discovery
│   ├── device/              # Device registry and management
│   └── config/              # Configuration management
├── proto/                   # Protocol buffer definitions
├── pkg/                     # Public API packages
└── examples/                # Usage examples
```

### Communication Protocol

The system uses gRPC for device-to-device communication with the following services:

- **RegisterDevice**: Register a device in the network
- **SendMessage**: Send messages between devices  
- **Heartbeat**: Maintain device status and health
- **GetDevices**: Retrieve list of available devices
- **SubmitTask**: Submit compute tasks for distribution
- **ExecuteSubtask**: Execute distributed subtasks
- **GetTaskStatus**: Monitor task progress

### Device Discovery (UDP-based)

Uses UDP broadcasts for automatic device discovery (inspired by EXO):
- Each device broadcasts presence every 2.5 seconds using UDP
- Discovers devices across WiFi, Ethernet, and cellular networks
- Automatic device capability detection and TFLOPS calculation
- Peer-to-peer architecture with no central coordinator required
- Works reliably across iOS, macOS, Linux, Windows, and Android

## Configuration

Configuration can be provided via:
- Command-line flags
- Environment variables
- YAML configuration file (`~/.discompute.yaml`)

Example configuration file:
```yaml
debug: true
port: 8080
tls: false
device-id: "my-custom-device-id"
```

## Development

### Building

```bash
# Generate protobuf files
make proto

# Build the CLI
make build

# Run tests
make test

# Clean build artifacts
make clean
```

### Cross-Platform Builds

```bash
# Build for macOS (Intel and Apple Silicon)
make build-darwin

# Build for iOS (requires iOS Go toolchain)
make build-ios
```

### Development Mode

Run the service in development mode with debug logging:

```bash
make dev-run
```

## Roadmap

### Milestone 2: Task Scheduling and Load Balancing
- [ ] Task scheduler implementation
- [ ] Device capability assessment
- [ ] Load balancing algorithms
- [ ] Fault tolerance and recovery

### Milestone 3: Task Execution Interface (Python)
- [ ] Python worker framework
- [ ] AI/ML framework integration
- [ ] Result aggregation
- [ ] Model parameter synchronization

### Milestone 4: Mobile Platform Support
- [ ] iOS application with Go runtime
- [ ] Android application integration
- [ ] Battery optimization
- [ ] Background processing

### Milestone 5: Security and Privacy
- [ ] TLS encryption for all communication
- [ ] Device authentication
- [ ] Access control policies
- [ ] Data privacy controls

## Contributing

This project is designed to be widely used in open source. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[License to be determined - likely MIT or Apache 2.0]

## Support

For questions and support:
- Create an issue on GitHub
- Join our community discussions
- Check the documentation

---

**Note**: This project is in active development. APIs and interfaces may change as we progress through the milestones.
