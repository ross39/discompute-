.PHONY: proto clean build test install ios android mobile deps

# Generate protobuf files
proto:
	@echo "Generating protobuf files..."
	@protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/device.proto

# Clean generated files and build artifacts
clean:
	@echo "Cleaning..."
	@rm -f proto/*.pb.go
	@rm -rf bin/
	@rm -rf mobile/
	@go clean

# Install dependencies
deps:
	@echo "Installing dependencies..."
	@go mod tidy
	@go mod download

# Build the CLI for current platform
build: proto deps
	@echo "Building discompute CLI..."
	@mkdir -p bin
	@go build -o bin/discompute ./cmd/discompute

# Build for specific platforms
build-darwin: proto deps
	@echo "Building for macOS..."
	@mkdir -p bin
	@GOOS=darwin GOARCH=amd64 go build -o bin/discompute-darwin-amd64 ./cmd/discompute
	@GOOS=darwin GOARCH=arm64 go build -o bin/discompute-darwin-arm64 ./cmd/discompute

build-linux: proto deps
	@echo "Building for Linux..."
	@mkdir -p bin
	@GOOS=linux GOARCH=amd64 go build -o bin/discompute-linux-amd64 ./cmd/discompute
	@GOOS=linux GOARCH=arm64 go build -o bin/discompute-linux-arm64 ./cmd/discompute

# iOS builds using gomobile
setup-mobile:
	@echo "Setting up mobile development..."
	@go install golang.org/x/mobile/cmd/gomobile@latest
	@gomobile init

ios: setup-mobile proto deps
	@echo "Building for iOS..."
	@mkdir -p mobile/ios
	@gomobile bind -target=ios -o mobile/ios/Discompute.xcframework ./mobile/

android: setup-mobile proto deps
	@echo "Building for Android..."
	@mkdir -p mobile/android
	@gomobile bind -target=android -o mobile/android/discompute.aar ./mobile/

# Build all mobile platforms
mobile: ios android

# Create iOS app project
ios-app: ios
	@echo "Creating iOS app project..."
	@mkdir -p ios-app
	@cp -r mobile/ios/Discompute.xcframework ios-app/
	@echo "iOS framework ready at ios-app/Discompute.xcframework"
	@echo "Import this framework in your Xcode project"

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Install the CLI
install: build
	@echo "Installing discompute CLI..."
	@cp bin/discompute $(GOPATH)/bin/ || cp bin/discompute /usr/local/bin/

# Development commands
dev-run: build
	@echo "Running discompute in development mode..."
	@./bin/discompute start --debug

# Check for required tools
check-tools:
	@echo "Checking for required tools..."
	@which protoc > /dev/null || (echo "protoc is required but not installed. Please install Protocol Buffers compiler." && exit 1)
	@which protoc-gen-go > /dev/null || (echo "protoc-gen-go is required. Run: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest" && exit 1)
	@which protoc-gen-go-grpc > /dev/null || (echo "protoc-gen-go-grpc is required. Run: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest" && exit 1)
	@echo "All required tools are installed."

# Setup development environment
setup: check-tools deps proto
	@echo "Development environment setup complete!"

# Build everything
all: setup build mobile

# Quick test on local network
test-local: build
	@echo "Starting test server..."
	@./bin/discompute start --debug --port 8080 &
	@sleep 3
	@echo "Testing device list..."
	@./bin/discompute list
	@echo "Stopping test server..."
	@pkill discompute || true