package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/rossheaney/discompute/internal/device"
	"github.com/rossheaney/discompute/internal/discovery"
	"github.com/rossheaney/discompute/internal/server"
	"github.com/rossheaney/discompute/internal/training"
	"github.com/sirupsen/logrus"
)

var (
	nodeID        = flag.String("node-id", "", "Node ID (auto-generated if empty)")
	nodePort      = flag.Int("node-port", 50051, "gRPC server port")
	httpPort      = flag.Int("http-port", 8080, "HTTP API port")
	listenPort    = flag.Int("listen-port", 5005, "UDP discovery listen port")
	broadcastPort = flag.Int("broadcast-port", 5005, "UDP discovery broadcast port")
	logLevel      = flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	mode          = flag.String("mode", "server", "Run mode: server, client, or training")

	// Training specific flags
	trainingModel  = flag.String("training-model", "mnist_cnn", "Training model type")
	trainingEpochs = flag.Int("training-epochs", 10, "Number of training epochs")
	trainingBatch  = flag.Int("training-batch", 32, "Training batch size")
	trainingLR     = flag.Float64("training-lr", 0.001, "Learning rate")
	maxDevices     = flag.Int("max-devices", 4, "Maximum devices for training")
)

func main() {
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	level, err := logrus.ParseLevel(*logLevel)
	if err != nil {
		logger.Fatal("Invalid log level:", err)
	}
	logger.SetLevel(level)

	// Generate node ID if not provided
	if *nodeID == "" {
		*nodeID = fmt.Sprintf("discompute_%s", uuid.New().String()[:8])
	}

	logger.WithFields(logrus.Fields{
		"node_id":   *nodeID,
		"mode":      *mode,
		"node_port": *nodePort,
		"http_port": *httpPort,
	}).Info("Starting Discompute")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	switch *mode {
	case "server":
		err = runServer(ctx, logger)
	case "client":
		err = runClient(ctx, logger)
	case "training":
		err = runTraining(ctx, logger)
	default:
		logger.Fatal("Invalid mode. Use: server, client, or training")
	}

	if err != nil {
		logger.Fatal("Error:", err)
	}

	// Wait for shutdown signal
	<-sigChan
	logger.Info("Shutdown signal received, stopping...")
	cancel()

	// Give services time to shutdown gracefully
	time.Sleep(2 * time.Second)
	logger.Info("Discompute stopped")
}

func runServer(ctx context.Context, logger *logrus.Logger) error {
	logger.Info("Starting in server mode...")

	// Create device registry
	registry := device.NewRegistry(logger)

	// Create UDP discovery
	udpDiscovery := discovery.NewUDPDiscovery(*nodeID, *nodePort, logger)
	udpDiscovery.SetPorts(*listenPort, *broadcastPort)

	// Set discovery callbacks
	udpDiscovery.SetCallbacks(
		func(discoveredDevice discovery.DiscoveredDevice) {
			logger.WithFields(logrus.Fields{
				"device_id":   discoveredDevice.ID,
				"device_name": discoveredDevice.Name,
				"device_type": discoveredDevice.Type,
				"address":     fmt.Sprintf("%s:%d", discoveredDevice.Address, discoveredDevice.Port),
			}).Info("Device discovered")

			// Convert to protobuf device and register
			// This is a simplified conversion - in practice you'd want proper mapping
			// registry.RegisterDevice(convertToProtoDevice(discoveredDevice))
		},
		func(deviceID string) {
			logger.WithField("device_id", deviceID).Info("Device lost")
			registry.UnregisterDevice(deviceID)
		},
	)

	// Start UDP discovery
	if err := udpDiscovery.Start(ctx); err != nil {
		return fmt.Errorf("failed to start UDP discovery: %w", err)
	}
	defer udpDiscovery.Stop()

	// Create gRPC server
	grpcServer := server.NewGRPCServer(registry, *nodePort, logger)

	// Register message handlers
	basicHandler := server.NewBasicMessageHandler(logger)
	grpcServer.RegisterMessageHandler("basic", basicHandler)

	// Start gRPC server
	if err := grpcServer.Start(); err != nil {
		return fmt.Errorf("failed to start gRPC server: %w", err)
	}
	defer grpcServer.Stop()

	// Create distributed trainer
	trainer := training.NewDistributedTrainer(registry, logger)
	_ = trainer // Used for future training endpoints

	logger.Info("Server started successfully")
	logger.WithFields(logrus.Fields{
		"grpc_port":      *nodePort,
		"udp_port":       *listenPort,
		"broadcast_port": *broadcastPort,
	}).Info("Listening for connections")

	// Keep running until context is cancelled
	<-ctx.Done()
	return nil
}

func runClient(ctx context.Context, logger *logrus.Logger) error {
	logger.Info("Starting in client mode...")

	// Create device registry (for local device info)
	registry := device.NewRegistry(logger)
	_ = registry // Used for device management

	// Create UDP discovery
	udpDiscovery := discovery.NewUDPDiscovery(*nodeID, *nodePort, logger)
	udpDiscovery.SetPorts(*listenPort, *broadcastPort)

	// Set discovery callbacks for client mode
	udpDiscovery.SetCallbacks(
		func(discoveredDevice discovery.DiscoveredDevice) {
			logger.WithFields(logrus.Fields{
				"server_id":   discoveredDevice.ID,
				"server_type": discoveredDevice.Type,
				"address":     fmt.Sprintf("%s:%d", discoveredDevice.Address, discoveredDevice.Port),
			}).Info("Server discovered")
		},
		func(deviceID string) {
			logger.WithField("server_id", deviceID).Info("Server lost")
		},
	)

	// Start UDP discovery
	if err := udpDiscovery.Start(ctx); err != nil {
		return fmt.Errorf("failed to start UDP discovery: %w", err)
	}
	defer udpDiscovery.Stop()

	logger.Info("Client started successfully")
	logger.Info("Broadcasting presence and looking for servers...")

	// Keep running until context is cancelled
	<-ctx.Done()
	return nil
}

func runTraining(ctx context.Context, logger *logrus.Logger) error {
	logger.Info("Starting distributed training demo...")

	// Create device registry
	registry := device.NewRegistry(logger)

	// Create UDP discovery
	udpDiscovery := discovery.NewUDPDiscovery(*nodeID, *nodePort, logger)
	udpDiscovery.SetPorts(*listenPort, *broadcastPort)

	// Track discovered devices
	discoveredDevices := make(map[string]discovery.DiscoveredDevice)

	udpDiscovery.SetCallbacks(
		func(discoveredDevice discovery.DiscoveredDevice) {
			discoveredDevices[discoveredDevice.ID] = discoveredDevice
			logger.WithFields(logrus.Fields{
				"device_id":     discoveredDevice.ID,
				"device_type":   discoveredDevice.Type,
				"total_devices": len(discoveredDevices),
			}).Info("Training device discovered")
		},
		func(deviceID string) {
			delete(discoveredDevices, deviceID)
			logger.WithField("device_id", deviceID).Info("Training device lost")
		},
	)

	// Start UDP discovery
	if err := udpDiscovery.Start(ctx); err != nil {
		return fmt.Errorf("failed to start UDP discovery: %w", err)
	}
	defer udpDiscovery.Stop()

	// Create distributed trainer
	trainer := training.NewDistributedTrainer(registry, logger)

	// Wait for devices to be discovered
	logger.Info("Waiting for Mac devices to join the training cluster...")
	logger.Info("Make sure your Mac devices are running the worker client!")

	waitTime := 30 * time.Second
	timer := time.NewTimer(waitTime)
	defer timer.Stop()

	select {
	case <-timer.C:
		logger.Info("Discovery period ended")
	case <-ctx.Done():
		return nil
	}

	// Check if we have enough devices
	macDevices := 0
	for _, device := range discoveredDevices {
		if device.Type == "mac" || device.Type == "macbook" || device.Type == "imac" || device.Type == "mac_mini" || device.Type == "mac_studio" {
			macDevices++
		}
	}

	if macDevices == 0 {
		logger.Warn("No Mac devices discovered. Make sure:")
		logger.Warn("  1. Mac devices are running discompute_mac_worker.py")
		logger.Warn("  2. All devices are on the same network")
		logger.Warn("  3. UDP port 5005 is not blocked")
		return fmt.Errorf("no Mac devices available for training")
	}

	logger.WithField("mac_devices", macDevices).Info("Found Mac devices, starting training...")

	// Create training configuration
	config := training.TrainingConfig{
		ModelType:        *trainingModel,
		BatchSize:        *trainingBatch,
		LearningRate:     *trainingLR,
		Epochs:           *trainingEpochs,
		DistributionMode: "data_parallel",
		Optimizer:        "adam",
		Parameters: map[string]interface{}{
			"input_size":  784, // MNIST 28x28
			"num_classes": 10,  // 10 digits
		},
	}

	// Start training job
	job, err := trainer.StartTrainingJob(ctx, config, *maxDevices)
	if err != nil {
		return fmt.Errorf("failed to start training job: %w", err)
	}

	logger.WithFields(logrus.Fields{
		"job_id":         job.JobID,
		"model_type":     job.Config.ModelType,
		"epochs":         job.Config.Epochs,
		"batch_size":     job.Config.BatchSize,
		"master_device":  job.MasterDevice,
		"worker_devices": len(job.WorkerDevices),
	}).Info("Distributed training job started!")

	// Monitor training progress
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				job, exists := trainer.GetTrainingJob(job.JobID)
				if !exists {
					return
				}

				logger.WithFields(logrus.Fields{
					"job_id":       job.JobID,
					"status":       job.Status,
					"epoch":        job.CurrentEpoch,
					"total_epochs": job.Config.Epochs,
					"avg_loss":     job.Metrics["avg_loss"],
					"avg_accuracy": job.Metrics["avg_accuracy"],
				}).Info("Training progress")

				if job.Status == "completed" || job.Status == "failed" {
					return
				}

			case <-ctx.Done():
				return
			}
		}
	}()

	logger.Info("Training in progress... Press Ctrl+C to stop")

	// Wait for completion or cancellation
	<-ctx.Done()

	// Stop training job
	if err := trainer.StopTrainingJob(job.JobID); err != nil {
		logger.WithError(err).Error("Failed to stop training job")
	}

	return nil
}
