package training

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/rossheaney/discompute/internal/device"
	pb "github.com/rossheaney/discompute/proto"
	"github.com/sirupsen/logrus"
)

// DistributedTrainer manages distributed neural network training across iOS devices
type DistributedTrainer struct {
	logger   *logrus.Logger
	registry *device.Registry

	// Training configuration
	config TrainingConfig

	// Device management
	mu               sync.RWMutex
	availableDevices map[string]*pb.Device
	activeTraining   map[string]*TrainingJob

	// Communication
	deviceClients map[string]DeviceClient
}

// TrainingConfig defines the training parameters
type TrainingConfig struct {
	ModelType        string                 `json:"model_type"` // "mnist_cnn", "simple_mlp", etc.
	BatchSize        int                    `json:"batch_size"`
	LearningRate     float64                `json:"learning_rate"`
	Epochs           int                    `json:"epochs"`
	DistributionMode string                 `json:"distribution_mode"` // "data_parallel", "model_parallel"
	Optimizer        string                 `json:"optimizer"`         // "adam", "sgd"
	Parameters       map[string]interface{} `json:"parameters"`
}

// TrainingJob represents an active distributed training job
type TrainingJob struct {
	JobID         string                 `json:"job_id"`
	Config        TrainingConfig         `json:"config"`
	MasterDevice  string                 `json:"master_device"`
	WorkerDevices []string               `json:"worker_devices"`
	Status        string                 `json:"status"`
	CurrentEpoch  int                    `json:"current_epoch"`
	StartTime     time.Time              `json:"start_time"`
	Metrics       map[string]float64     `json:"metrics"`
	ModelState    map[string]interface{} `json:"model_state"`
}

// DeviceClient interface for communicating with iOS devices
type DeviceClient interface {
	SendTrainingBatch(ctx context.Context, batch *TrainingBatch) (*TrainingResult, error)
	SendModelUpdate(ctx context.Context, update *ModelUpdate) error
	GetDeviceMetrics(ctx context.Context) (*DeviceMetrics, error)
	InitializeTraining(ctx context.Context, config *TrainingConfig) error
	Shutdown(ctx context.Context) error
}

// TrainingBatch represents a batch of training data sent to a device
type TrainingBatch struct {
	BatchID  string                 `json:"batch_id"`
	JobID    string                 `json:"job_id"`
	Epoch    int                    `json:"epoch"`
	BatchIdx int                    `json:"batch_idx"`
	Data     [][]float64            `json:"data"`   // Input data
	Labels   []int                  `json:"labels"` // Target labels
	Metadata map[string]interface{} `json:"metadata"`
}

// TrainingResult represents the result from training a batch
type TrainingResult struct {
	BatchID        string               `json:"batch_id"`
	Loss           float64              `json:"loss"`
	Accuracy       float64              `json:"accuracy"`
	Gradients      map[string][]float64 `json:"gradients"`
	ProcessingTime time.Duration        `json:"processing_time"`
	DeviceMetrics  *DeviceMetrics       `json:"device_metrics"`
}

// ModelUpdate represents a model parameter update
type ModelUpdate struct {
	UpdateID   string               `json:"update_id"`
	JobID      string               `json:"job_id"`
	Epoch      int                  `json:"epoch"`
	Parameters map[string][]float64 `json:"parameters"`
	UpdateType string               `json:"update_type"` // "gradient", "parameter"
}

// DeviceMetrics represents device performance metrics during training
type DeviceMetrics struct {
	CPUUsage      float64 `json:"cpu_usage"`
	MemoryUsage   float64 `json:"memory_usage"`
	BatteryLevel  float64 `json:"battery_level"`
	Temperature   float64 `json:"temperature"`
	ThroughputTPS float64 `json:"throughput_tps"` // Tensors per second
}

// NewDistributedTrainer creates a new distributed trainer
func NewDistributedTrainer(registry *device.Registry, logger *logrus.Logger) *DistributedTrainer {
	return &DistributedTrainer{
		logger:           logger,
		registry:         registry,
		availableDevices: make(map[string]*pb.Device),
		activeTraining:   make(map[string]*TrainingJob),
		deviceClients:    make(map[string]DeviceClient),
	}
}

// StartTrainingJob starts a new distributed training job
func (dt *DistributedTrainer) StartTrainingJob(ctx context.Context, config TrainingConfig, maxDevices int) (*TrainingJob, error) {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	// Generate job ID
	jobID := fmt.Sprintf("job_%d", time.Now().Unix())

	// Select available iOS devices
	devices := dt.selectDevicesForTraining(maxDevices)
	if len(devices) == 0 {
		return nil, fmt.Errorf("no available iOS devices for training")
	}

	// Create training job
	job := &TrainingJob{
		JobID:         jobID,
		Config:        config,
		MasterDevice:  devices[0], // First device is master
		WorkerDevices: devices[1:],
		Status:        "initializing",
		CurrentEpoch:  0,
		StartTime:     time.Now(),
		Metrics:       make(map[string]float64),
		ModelState:    make(map[string]interface{}),
	}

	dt.activeTraining[jobID] = job

	dt.logger.WithFields(logrus.Fields{
		"job_id":         jobID,
		"master_device":  job.MasterDevice,
		"worker_devices": len(job.WorkerDevices),
		"total_devices":  len(devices),
		"model_type":     config.ModelType,
	}).Info("Starting distributed training job")

	// Initialize training on all devices
	go dt.runTrainingJob(ctx, job)

	return job, nil
}

// selectDevicesForTraining selects the best available iOS devices for training
func (dt *DistributedTrainer) selectDevicesForTraining(maxDevices int) []string {
	allDevices := dt.registry.GetAllDevices()
	var iosDevices []*pb.Device

	// Filter for available iOS devices
	for _, device := range allDevices {
		if (device.Type == "iphone" || device.Type == "ipad") &&
			device.Status == pb.DeviceStatus_AVAILABLE &&
			device.Capabilities.BatteryLevel > 0.2 { // At least 20% battery
			iosDevices = append(iosDevices, device)
		}
	}

	// Sort by performance (simplified scoring)
	// In practice, you'd want a more sophisticated scoring algorithm
	deviceIDs := make([]string, 0, len(iosDevices))
	for i, device := range iosDevices {
		if i >= maxDevices {
			break
		}
		deviceIDs = append(deviceIDs, device.Id)
	}

	return deviceIDs
}

// runTrainingJob executes the distributed training job
func (dt *DistributedTrainer) runTrainingJob(ctx context.Context, job *TrainingJob) {
	defer func() {
		dt.mu.Lock()
		job.Status = "completed"
		dt.mu.Unlock()
	}()

	dt.logger.WithField("job_id", job.JobID).Info("Initializing training on devices")

	// Initialize training on all devices
	if err := dt.initializeDevicesForTraining(ctx, job); err != nil {
		dt.logger.WithError(err).Error("Failed to initialize devices for training")
		job.Status = "failed"
		return
	}

	job.Status = "training"

	// Load training data (for MNIST demo)
	trainingData, err := dt.loadTrainingData(job.Config.ModelType)
	if err != nil {
		dt.logger.WithError(err).Error("Failed to load training data")
		job.Status = "failed"
		return
	}

	// Run training epochs
	for epoch := 0; epoch < job.Config.Epochs; epoch++ {
		dt.logger.WithFields(logrus.Fields{
			"job_id": job.JobID,
			"epoch":  epoch + 1,
			"total":  job.Config.Epochs,
		}).Info("Starting training epoch")

		job.CurrentEpoch = epoch

		if err := dt.runEpoch(ctx, job, trainingData, epoch); err != nil {
			dt.logger.WithError(err).Error("Training epoch failed")
			job.Status = "failed"
			return
		}

		// Update metrics
		dt.updateJobMetrics(job)
	}

	dt.logger.WithField("job_id", job.JobID).Info("Distributed training completed successfully")
}

// initializeDevicesForTraining initializes all devices for the training job
func (dt *DistributedTrainer) initializeDevicesForTraining(ctx context.Context, job *TrainingJob) error {
	allDevices := append([]string{job.MasterDevice}, job.WorkerDevices...)

	for _, deviceID := range allDevices {
		client, exists := dt.deviceClients[deviceID]
		if !exists {
			// Create new device client
			device, exists := dt.registry.GetDevice(deviceID)
			if !exists {
				return fmt.Errorf("device %s not found in registry", deviceID)
			}

			client = NewIOSDeviceClient(device, dt.logger)
			dt.deviceClients[deviceID] = client
		}

		if err := client.InitializeTraining(ctx, &job.Config); err != nil {
			return fmt.Errorf("failed to initialize training on device %s: %w", deviceID, err)
		}
	}

	return nil
}

// loadTrainingData loads the training dataset
func (dt *DistributedTrainer) loadTrainingData(modelType string) (interface{}, error) {
	switch modelType {
	case "mnist_cnn":
		// For now, return placeholder data
		// In a real implementation, you'd load actual MNIST data
		return map[string]interface{}{
			"type":       "mnist",
			"batches":    100,
			"batch_size": 32,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported model type: %s", modelType)
	}
}

// runEpoch runs a single training epoch across all devices
func (dt *DistributedTrainer) runEpoch(ctx context.Context, job *TrainingJob, trainingData interface{}, epoch int) error {
	// For data parallel training, distribute batches across devices
	allDevices := append([]string{job.MasterDevice}, job.WorkerDevices...)

	// Create batches for this epoch
	batches := dt.createTrainingBatches(job, trainingData, epoch)

	// Distribute batches across devices
	batchesPerDevice := len(batches) / len(allDevices)
	if batchesPerDevice == 0 {
		batchesPerDevice = 1
	}

	var wg sync.WaitGroup
	results := make(chan *TrainingResult, len(batches))
	errors := make(chan error, len(allDevices))

	// Process batches on each device
	for i, deviceID := range allDevices {
		wg.Add(1)
		go func(deviceID string, startIdx int) {
			defer wg.Done()

			endIdx := startIdx + batchesPerDevice
			if endIdx > len(batches) {
				endIdx = len(batches)
			}

			client := dt.deviceClients[deviceID]
			for j := startIdx; j < endIdx; j++ {
				result, err := client.SendTrainingBatch(ctx, batches[j])
				if err != nil {
					errors <- fmt.Errorf("device %s batch %d failed: %w", deviceID, j, err)
					return
				}
				results <- result
			}
		}(deviceID, i*batchesPerDevice)
	}

	wg.Wait()
	close(results)
	close(errors)

	// Check for errors
	select {
	case err := <-errors:
		return err
	default:
	}

	// Aggregate results and update model
	return dt.aggregateAndUpdateModel(job, results)
}

// createTrainingBatches creates training batches for an epoch
func (dt *DistributedTrainer) createTrainingBatches(job *TrainingJob, trainingData interface{}, epoch int) []*TrainingBatch {
	// Simplified batch creation for demo
	// In practice, you'd create actual data batches from your dataset

	numBatches := 10 // Simplified for demo
	batches := make([]*TrainingBatch, numBatches)

	for i := 0; i < numBatches; i++ {
		batches[i] = &TrainingBatch{
			BatchID:  fmt.Sprintf("%s_e%d_b%d", job.JobID, epoch, i),
			JobID:    job.JobID,
			Epoch:    epoch,
			BatchIdx: i,
			Data:     generateDummyMNISTBatch(job.Config.BatchSize), // Placeholder
			Labels:   generateDummyLabels(job.Config.BatchSize),     // Placeholder
			Metadata: map[string]interface{}{
				"epoch": epoch,
				"batch": i,
			},
		}
	}

	return batches
}

// aggregateAndUpdateModel aggregates training results and updates the model
func (dt *DistributedTrainer) aggregateAndUpdateModel(job *TrainingJob, results <-chan *TrainingResult) error {
	var totalLoss, totalAccuracy float64
	var resultCount int

	// Aggregate gradients (simplified)
	aggregatedGradients := make(map[string][]float64)

	for result := range results {
		totalLoss += result.Loss
		totalAccuracy += result.Accuracy
		resultCount++

		// Aggregate gradients
		for layer, grads := range result.Gradients {
			if existing, exists := aggregatedGradients[layer]; exists {
				// Average gradients
				for i, grad := range grads {
					if i < len(existing) {
						existing[i] = (existing[i] + grad) / 2.0
					}
				}
			} else {
				aggregatedGradients[layer] = grads
			}
		}
	}

	if resultCount > 0 {
		job.Metrics["avg_loss"] = totalLoss / float64(resultCount)
		job.Metrics["avg_accuracy"] = totalAccuracy / float64(resultCount)
	}

	// Send model updates to all devices
	update := &ModelUpdate{
		UpdateID:   fmt.Sprintf("%s_update_%d", job.JobID, job.CurrentEpoch),
		JobID:      job.JobID,
		Epoch:      job.CurrentEpoch,
		Parameters: aggregatedGradients,
		UpdateType: "gradient",
	}

	return dt.broadcastModelUpdate(context.Background(), job, update)
}

// broadcastModelUpdate sends model updates to all devices
func (dt *DistributedTrainer) broadcastModelUpdate(ctx context.Context, job *TrainingJob, update *ModelUpdate) error {
	allDevices := append([]string{job.MasterDevice}, job.WorkerDevices...)

	for _, deviceID := range allDevices {
		client := dt.deviceClients[deviceID]
		if err := client.SendModelUpdate(ctx, update); err != nil {
			dt.logger.WithError(err).WithField("device_id", deviceID).Error("Failed to send model update")
			// Continue with other devices rather than failing completely
		}
	}

	return nil
}

// updateJobMetrics updates job metrics from device metrics
func (dt *DistributedTrainer) updateJobMetrics(job *TrainingJob) {
	// Collect metrics from all devices
	allDevices := append([]string{job.MasterDevice}, job.WorkerDevices...)

	var totalCPU, totalMemory, totalBattery float64
	var deviceCount int

	for _, deviceID := range allDevices {
		client := dt.deviceClients[deviceID]
		metrics, err := client.GetDeviceMetrics(context.Background())
		if err != nil {
			dt.logger.WithError(err).WithField("device_id", deviceID).Debug("Failed to get device metrics")
			continue
		}

		totalCPU += metrics.CPUUsage
		totalMemory += metrics.MemoryUsage
		totalBattery += metrics.BatteryLevel
		deviceCount++
	}

	if deviceCount > 0 {
		job.Metrics["avg_cpu_usage"] = totalCPU / float64(deviceCount)
		job.Metrics["avg_memory_usage"] = totalMemory / float64(deviceCount)
		job.Metrics["avg_battery_level"] = totalBattery / float64(deviceCount)
	}
}

// GetTrainingJob returns information about a training job
func (dt *DistributedTrainer) GetTrainingJob(jobID string) (*TrainingJob, bool) {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	job, exists := dt.activeTraining[jobID]
	return job, exists
}

// ListActiveJobs returns all active training jobs
func (dt *DistributedTrainer) ListActiveJobs() []*TrainingJob {
	dt.mu.RLock()
	defer dt.mu.RUnlock()

	jobs := make([]*TrainingJob, 0, len(dt.activeTraining))
	for _, job := range dt.activeTraining {
		jobs = append(jobs, job)
	}

	return jobs
}

// StopTrainingJob stops a training job
func (dt *DistributedTrainer) StopTrainingJob(jobID string) error {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	job, exists := dt.activeTraining[jobID]
	if !exists {
		return fmt.Errorf("training job %s not found", jobID)
	}

	job.Status = "stopping"

	// Shutdown devices
	allDevices := append([]string{job.MasterDevice}, job.WorkerDevices...)
	for _, deviceID := range allDevices {
		if client, exists := dt.deviceClients[deviceID]; exists {
			client.Shutdown(context.Background())
		}
	}

	delete(dt.activeTraining, jobID)

	dt.logger.WithField("job_id", jobID).Info("Training job stopped")
	return nil
}

// Helper functions for demo data generation
func generateDummyMNISTBatch(batchSize int) [][]float64 {
	batch := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		// 28x28 = 784 pixels for MNIST
		pixels := make([]float64, 784)
		for j := 0; j < 784; j++ {
			pixels[j] = float64(i%10) / 10.0 // Dummy data
		}
		batch[i] = pixels
	}
	return batch
}

func generateDummyLabels(batchSize int) []int {
	labels := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		labels[i] = i % 10 // Digits 0-9
	}
	return labels
}
