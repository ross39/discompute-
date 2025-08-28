package main

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/rossheaney/discompute/internal/client"
	"github.com/rossheaney/discompute/internal/device"
	"github.com/rossheaney/discompute/internal/discovery"
	"github.com/rossheaney/discompute/internal/server"
	pb "github.com/rossheaney/discompute/proto"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	logger = logrus.New()

	// Global configuration
	cfgFile   string
	debug     bool
	port      int
	deviceID  string
	enableTLS bool
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "discompute",
	Short: "Distributed compute system for cross-device AI/ML workloads",
	Long: `discompute is a distributed compute system that enables AI/ML workloads 
to be distributed across multiple devices including MacBooks, iPhones, iPads, and other platforms.

This tool handles device discovery via UDP broadcasts, task scheduling, and secure communication 
between devices in the network. Based on peer-to-peer architecture inspired by EXO.`,
}

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the discompute service",
	Long: `Start the discompute service which will:
- Advertise this device on the local network via UDP broadcasts
- Discover other discompute devices using UDP discovery  
- Start the gRPC server for communication
- Begin accepting compute tasks`,
	Run: func(cmd *cobra.Command, args []string) {
		runStart()
	},
}

// sendCmd represents the send command for testing
var sendCmd = &cobra.Command{
	Use:   "send [target-device-id] [message]",
	Short: "Send a test message to another device",
	Long:  `Send a test text message to another device for testing connectivity.`,
	Args:  cobra.ExactArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		targetDeviceID := args[0]
		message := args[1]
		runSend(targetDeviceID, message)
	},
}

// listCmd represents the list command
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List discovered devices",
	Long:  `List all devices discovered on the local network via UDP broadcasts.`,
	Run: func(cmd *cobra.Command, args []string) {
		runList()
	},
}

// infoCmd shows device information
var infoCmd = &cobra.Command{
	Use:   "info",
	Short: "Show this device's capabilities",
	Long:  `Display detailed information about this device's compute capabilities.`,
	Run: func(cmd *cobra.Command, args []string) {
		runInfo()
	},
}

func init() {
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.discompute.yaml)")
	rootCmd.PersistentFlags().BoolVar(&debug, "debug", false, "enable debug logging")
	rootCmd.PersistentFlags().IntVar(&port, "port", 8080, "port to listen on")
	rootCmd.PersistentFlags().StringVar(&deviceID, "device-id", "", "unique device identifier (auto-generated if not provided)")
	rootCmd.PersistentFlags().BoolVar(&enableTLS, "tls", false, "enable TLS encryption")

	// Bind flags to viper
	viper.BindPFlag("debug", rootCmd.PersistentFlags().Lookup("debug"))
	viper.BindPFlag("port", rootCmd.PersistentFlags().Lookup("port"))
	viper.BindPFlag("device-id", rootCmd.PersistentFlags().Lookup("device-id"))
	viper.BindPFlag("tls", rootCmd.PersistentFlags().Lookup("tls"))

	// Add subcommands
	rootCmd.AddCommand(startCmd)
	rootCmd.AddCommand(sendCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(infoCmd)
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		viper.AddConfigPath(home)
		viper.SetConfigType("yaml")
		viper.SetConfigName(".discompute")
	}

	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		logger.WithField("config", viper.ConfigFileUsed()).Info("Using config file")
	}

	// Configure logging
	if viper.GetBool("debug") || debug {
		logger.SetLevel(logrus.DebugLevel)
		logger.Debug("Debug logging enabled")
	}
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		logger.WithError(err).Fatal("Command execution failed")
	}
}

func runStart() {
	// Generate device ID if not provided
	if deviceID == "" {
		deviceID = viper.GetString("device-id")
		if deviceID == "" {
			deviceID = generateDeviceID()
			logger.WithField("device_id", deviceID).Info("Generated device ID")
		}
	}

	// Get port from config
	if port == 8080 {
		port = viper.GetInt("port")
		if port == 0 {
			port = 8080
		}
	}

	// Get TLS setting
	if !enableTLS {
		enableTLS = viper.GetBool("tls")
	}

	logger.WithFields(logrus.Fields{
		"device_id": deviceID,
		"port":      port,
		"tls":       enableTLS,
	}).Info("Starting discompute service")

	// Show device capabilities
	caps := device.GetDeviceCapabilities(logger)
	logger.WithFields(logrus.Fields{
		"model":  caps.Model,
		"chip":   caps.Chip,
		"type":   caps.Type,
		"memory": fmt.Sprintf("%d MB", caps.Memory),
		"fp32":   fmt.Sprintf("%.2f TFLOPS", caps.Flops.FP32),
		"fp16":   fmt.Sprintf("%.2f TFLOPS", caps.Flops.FP16),
		"int8":   fmt.Sprintf("%.2f TFLOPS", caps.Flops.INT8),
	}).Info("Device capabilities detected")

	// Create device registry
	registry := device.NewRegistry(logger)

	// Create and start gRPC server
	grpcServer := server.NewGRPCServer(registry, port, logger)

	// Register basic message handler
	basicHandler := server.NewBasicMessageHandler(logger)
	grpcServer.RegisterMessageHandler("text", basicHandler)

	if err := grpcServer.Start(); err != nil {
		logger.WithError(err).Fatal("Failed to start gRPC server")
	}
	defer grpcServer.Stop()

	// Create UDP discovery service
	udpDiscovery := discovery.NewUDPDiscovery(deviceID, port, logger)

	// Set discovery callbacks
	udpDiscovery.SetCallbacks(
		func(deviceInfo discovery.DiscoveredDevice) {
			// Device found callback
			registry.RegisterDeviceFromDiscovery(deviceInfo)
		},
		func(deviceID string) {
			// Device lost callback
			registry.UnregisterDevice(deviceID)
		},
	)

	// Start discovery service
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := udpDiscovery.Start(ctx); err != nil {
		logger.WithError(err).Error("Failed to start UDP discovery, continuing without it")
	} else {
		defer udpDiscovery.Stop()
	}

	// Register this device in the registry
	thisDevice := &pb.Device{
		Id:      deviceID,
		Name:    caps.Model,
		Type:    caps.Type,
		Address: fmt.Sprintf("localhost:%d", port),
		Capabilities: &pb.DeviceCapabilities{
			CpuCores:     int32(runtime.NumCPU()),
			MemoryMb:     caps.Memory,
			HasGpu:       caps.Flops.FP16 > 0,
			GpuType:      caps.Chip,
			BatteryLevel: -1,
			IsCharging:   false,
			Fp32Tflops:   caps.Flops.FP32,
			Fp16Tflops:   caps.Flops.FP16,
			Int8Tflops:   caps.Flops.INT8,
			Chip:         caps.Chip,
		},
		Status: pb.DeviceStatus_AVAILABLE,
	}
	registry.RegisterDevice(thisDevice)

	// Start cleanup routine for stale devices
	go func() {
		ticker := time.NewTicker(1 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				registry.CleanupStaleDevices(3 * time.Minute)
			}
		}
	}()

	logger.Info("discompute service is running and ready for distributed compute tasks")

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	logger.Info("Shutting down discompute service")
}

func runSend(targetDeviceID, message string) {
	// Generate device ID if not provided
	if deviceID == "" {
		deviceID = generateDeviceID()
	}

	// Connect to local server
	serverAddr := fmt.Sprintf("localhost:%d", viper.GetInt("port"))
	if viper.GetInt("port") == 0 {
		serverAddr = "localhost:8080"
	}

	grpcClient := client.NewGRPCClient(serverAddr, viper.GetBool("tls"), logger)

	if err := grpcClient.Connect(); err != nil {
		logger.WithError(err).Fatal("Failed to connect to discompute service")
	}
	defer grpcClient.Disconnect()

	// Send the message
	err := grpcClient.SendTextMessage(deviceID, targetDeviceID, message)
	if err != nil {
		logger.WithError(err).Fatal("Failed to send message")
	}

	logger.WithFields(logrus.Fields{
		"target":  targetDeviceID,
		"message": message,
	}).Info("Message sent successfully")
}

func runList() {
	// Generate device ID if not provided
	if deviceID == "" {
		deviceID = generateDeviceID()
	}

	// Connect to local server
	serverAddr := fmt.Sprintf("localhost:%d", viper.GetInt("port"))
	if viper.GetInt("port") == 0 {
		serverAddr = "localhost:8080"
	}

	grpcClient := client.NewGRPCClient(serverAddr, viper.GetBool("tls"), logger)

	if err := grpcClient.Connect(); err != nil {
		logger.WithError(err).Fatal("Failed to connect to discompute service")
	}
	defer grpcClient.Disconnect()

	// Get devices list
	resp, err := grpcClient.GetDevices(deviceID)
	if err != nil {
		logger.WithError(err).Fatal("Failed to get devices list")
	}

	if len(resp.Devices) == 0 {
		fmt.Println("No devices found")
		return
	}

	fmt.Printf("Found %d device(s):\n\n", len(resp.Devices))

	for _, device := range resp.Devices {
		fmt.Printf("Device ID: %s\n", device.Id)
		fmt.Printf("  Name: %s\n", device.Name)
		fmt.Printf("  Type: %s\n", device.Type)
		fmt.Printf("  Address: %s\n", device.Address)
		fmt.Printf("  Status: %s\n", device.Status.String())
		if device.Capabilities != nil {
			fmt.Printf("  Capabilities:\n")
			fmt.Printf("    CPU Cores: %d\n", device.Capabilities.CpuCores)
			fmt.Printf("    Memory: %d MB\n", device.Capabilities.MemoryMb)
			fmt.Printf("    GPU: %t (%s)\n", device.Capabilities.HasGpu, device.Capabilities.GpuType)
			if device.Capabilities.Fp32Tflops > 0 {
				fmt.Printf("    Compute Performance:\n")
				fmt.Printf("      FP32: %.2f TFLOPS\n", device.Capabilities.Fp32Tflops)
				fmt.Printf("      FP16: %.2f TFLOPS\n", device.Capabilities.Fp16Tflops)
				fmt.Printf("      INT8: %.2f TFLOPS\n", device.Capabilities.Int8Tflops)
			}
			if device.Capabilities.BatteryLevel >= 0 {
				fmt.Printf("    Battery: %.1f%%", device.Capabilities.BatteryLevel*100)
				if device.Capabilities.IsCharging {
					fmt.Printf(" (charging)")
				}
				fmt.Println()
			}
		}
		fmt.Printf("  Last Seen: %s\n", time.Unix(device.LastSeen, 0).Format(time.RFC3339))
		fmt.Println()
	}
}

func runInfo() {
	caps := device.GetDeviceCapabilities(logger)

	fmt.Printf("Device Information:\n\n")
	fmt.Printf("Model: %s\n", caps.Model)
	fmt.Printf("Chip: %s\n", caps.Chip)
	fmt.Printf("Type: %s\n", caps.Type)
	fmt.Printf("Memory: %d MB\n", caps.Memory)
	fmt.Printf("\nCompute Performance:\n")
	fmt.Printf("  FP32: %.2f TFLOPS\n", caps.Flops.FP32)
	fmt.Printf("  FP16: %.2f TFLOPS\n", caps.Flops.FP16)
	fmt.Printf("  INT8: %.2f TFLOPS\n", caps.Flops.INT8)
	fmt.Printf("\nPlatform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPU Cores: %d\n", runtime.NumCPU())

	// Show what this device would be good for
	fmt.Printf("\nRecommended Use Cases:\n")
	if caps.Flops.FP32 > 10.0 {
		fmt.Printf("  ✓ High-performance training tasks\n")
		fmt.Printf("  ✓ Large model inference\n")
	} else if caps.Flops.FP32 > 2.0 {
		fmt.Printf("  ✓ Medium-scale training tasks\n")
		fmt.Printf("  ✓ Model inference\n")
	} else {
		fmt.Printf("  ✓ Data preprocessing\n")
		fmt.Printf("  ✓ Small model inference\n")
	}

	if caps.Type == "iphone" || caps.Type == "ipad" {
		fmt.Printf("  ✓ Mobile AI applications\n")
		fmt.Printf("  ✓ Edge computing tasks\n")
	}
}

func generateDeviceID() string {
	bytes := make([]byte, 8)
	rand.Read(bytes)
	return fmt.Sprintf("%x", bytes)
}

func getDeviceName() string {
	if name := os.Getenv("DEVICE_NAME"); name != "" {
		return name
	}

	hostname, err := os.Hostname()
	if err != nil {
		return "discompute-device"
	}

	return hostname
}

func getDeviceType() string {
	switch runtime.GOOS {
	case "darwin":
		// Try to determine if it's iOS or macOS
		if strings.Contains(runtime.GOARCH, "arm") {
			// Could be iOS or Apple Silicon Mac
			// For now, assume macOS - this would need more sophisticated detection
			return "mac"
		}
		return "mac"
	case "linux":
		return "linux"
	case "windows":
		return "windows"
	case "android":
		return "android"
	case "ios":
		return "iphone"
	default:
		return "unknown"
	}
}
