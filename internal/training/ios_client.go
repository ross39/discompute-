package training

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pb "github.com/rossheaney/discompute/proto"
	"github.com/sirupsen/logrus"
)

// IOSDeviceClient implements DeviceClient for iOS devices
type IOSDeviceClient struct {
	device     *pb.Device
	logger     *logrus.Logger
	httpClient *http.Client
	baseURL    string
}

// NewIOSDeviceClient creates a new iOS device client
func NewIOSDeviceClient(device *pb.Device, logger *logrus.Logger) *IOSDeviceClient {
	return &IOSDeviceClient{
		device: device,
		logger: logger,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		baseURL: fmt.Sprintf("http://%s", device.Address),
	}
}

// InitializeTraining initializes training on the iOS device
func (c *IOSDeviceClient) InitializeTraining(ctx context.Context, config *TrainingConfig) error {
	c.logger.WithFields(logrus.Fields{
		"device_id":  c.device.Id,
		"model_type": config.ModelType,
	}).Info("Initializing training on iOS device")

	payload := map[string]interface{}{
		"action": "initialize_training",
		"config": config,
	}

	response, err := c.sendRequest(ctx, "/api/training/initialize", payload)
	if err != nil {
		return fmt.Errorf("failed to initialize training: %w", err)
	}

	if !response["success"].(bool) {
		return fmt.Errorf("training initialization failed: %s", response["message"])
	}

	return nil
}

// SendTrainingBatch sends a training batch to the iOS device
func (c *IOSDeviceClient) SendTrainingBatch(ctx context.Context, batch *TrainingBatch) (*TrainingResult, error) {
	c.logger.WithFields(logrus.Fields{
		"device_id": c.device.Id,
		"batch_id":  batch.BatchID,
		"epoch":     batch.Epoch,
	}).Debug("Sending training batch to iOS device")

	payload := map[string]interface{}{
		"action": "train_batch",
		"batch":  batch,
	}

	response, err := c.sendRequest(ctx, "/api/training/batch", payload)
	if err != nil {
		return nil, fmt.Errorf("failed to send training batch: %w", err)
	}

	if !response["success"].(bool) {
		return nil, fmt.Errorf("training batch failed: %s", response["message"])
	}

	// Parse training result
	resultData, ok := response["result"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid training result format")
	}

	result := &TrainingResult{
		BatchID:  batch.BatchID,
		Loss:     resultData["loss"].(float64),
		Accuracy: resultData["accuracy"].(float64),
	}

	// Parse gradients if present
	if grads, exists := resultData["gradients"].(map[string]interface{}); exists {
		result.Gradients = make(map[string][]float64)
		for layer, gradData := range grads {
			if gradSlice, ok := gradData.([]interface{}); ok {
				gradients := make([]float64, len(gradSlice))
				for i, v := range gradSlice {
					gradients[i] = v.(float64)
				}
				result.Gradients[layer] = gradients
			}
		}
	}

	// Parse device metrics if present
	if metricsData, exists := resultData["device_metrics"].(map[string]interface{}); exists {
		result.DeviceMetrics = &DeviceMetrics{
			CPUUsage:      metricsData["cpu_usage"].(float64),
			MemoryUsage:   metricsData["memory_usage"].(float64),
			BatteryLevel:  metricsData["battery_level"].(float64),
			Temperature:   metricsData["temperature"].(float64),
			ThroughputTPS: metricsData["throughput_tps"].(float64),
		}
	}

	return result, nil
}

// SendModelUpdate sends a model update to the iOS device
func (c *IOSDeviceClient) SendModelUpdate(ctx context.Context, update *ModelUpdate) error {
	c.logger.WithFields(logrus.Fields{
		"device_id": c.device.Id,
		"update_id": update.UpdateID,
		"epoch":     update.Epoch,
	}).Debug("Sending model update to iOS device")

	payload := map[string]interface{}{
		"action": "model_update",
		"update": update,
	}

	response, err := c.sendRequest(ctx, "/api/training/update", payload)
	if err != nil {
		return fmt.Errorf("failed to send model update: %w", err)
	}

	if !response["success"].(bool) {
		return fmt.Errorf("model update failed: %s", response["message"])
	}

	return nil
}

// GetDeviceMetrics retrieves current device metrics
func (c *IOSDeviceClient) GetDeviceMetrics(ctx context.Context) (*DeviceMetrics, error) {
	payload := map[string]interface{}{
		"action": "get_metrics",
	}

	response, err := c.sendRequest(ctx, "/api/device/metrics", payload)
	if err != nil {
		return nil, fmt.Errorf("failed to get device metrics: %w", err)
	}

	if !response["success"].(bool) {
		return nil, fmt.Errorf("failed to get metrics: %s", response["message"])
	}

	metricsData, ok := response["metrics"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid metrics format")
	}

	metrics := &DeviceMetrics{
		CPUUsage:      metricsData["cpu_usage"].(float64),
		MemoryUsage:   metricsData["memory_usage"].(float64),
		BatteryLevel:  metricsData["battery_level"].(float64),
		Temperature:   metricsData["temperature"].(float64),
		ThroughputTPS: metricsData["throughput_tps"].(float64),
	}

	return metrics, nil
}

// Shutdown gracefully shuts down training on the iOS device
func (c *IOSDeviceClient) Shutdown(ctx context.Context) error {
	c.logger.WithField("device_id", c.device.Id).Info("Shutting down training on iOS device")

	payload := map[string]interface{}{
		"action": "shutdown_training",
	}

	response, err := c.sendRequest(ctx, "/api/training/shutdown", payload)
	if err != nil {
		return fmt.Errorf("failed to shutdown training: %w", err)
	}

	if !response["success"].(bool) {
		return fmt.Errorf("shutdown failed: %s", response["message"])
	}

	return nil
}

// sendRequest sends an HTTP request to the iOS device
func (c *IOSDeviceClient) sendRequest(ctx context.Context, endpoint string, payload map[string]interface{}) (map[string]interface{}, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "discompute-client/1.0")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return response, nil
}

// HealthCheck performs a health check on the iOS device
func (c *IOSDeviceClient) HealthCheck(ctx context.Context) error {
	payload := map[string]interface{}{
		"action": "health_check",
	}

	response, err := c.sendRequest(ctx, "/api/health", payload)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	if !response["success"].(bool) {
		return fmt.Errorf("device unhealthy: %s", response["message"])
	}

	return nil
}
