package client

import (
	"context"
	"crypto/tls"
	"fmt"
	"time"

	pb "github.com/rossheaney/discompute/proto"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// GRPCClient wraps the gRPC client for device communication
type GRPCClient struct {
	conn   *grpc.ClientConn
	client pb.DeviceServiceClient
	logger *logrus.Logger

	// Client configuration
	serverAddr string
	enableTLS  bool
	timeout    time.Duration
}

// NewGRPCClient creates a new gRPC client
func NewGRPCClient(serverAddr string, enableTLS bool, logger *logrus.Logger) *GRPCClient {
	return &GRPCClient{
		serverAddr: serverAddr,
		enableTLS:  enableTLS,
		timeout:    30 * time.Second,
		logger:     logger,
	}
}

// SetTimeout sets the request timeout
func (c *GRPCClient) SetTimeout(timeout time.Duration) {
	c.timeout = timeout
}

// Connect establishes a connection to the gRPC server
func (c *GRPCClient) Connect() error {
	var opts []grpc.DialOption

	// Configure TLS or insecure connection
	if c.enableTLS {
		config := &tls.Config{
			ServerName: "localhost", // This should be configurable for production
		}
		creds := credentials.NewTLS(config)
		opts = append(opts, grpc.WithTransportCredentials(creds))
		c.logger.Info("Using TLS for gRPC connection")
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	c.logger.WithField("server_addr", c.serverAddr).Info("Connecting to gRPC server")

	conn, err := grpc.Dial(c.serverAddr, opts...)
	if err != nil {
		return fmt.Errorf("failed to connect to gRPC server: %w", err)
	}

	c.conn = conn
	c.client = pb.NewDeviceServiceClient(conn)

	c.logger.Info("Connected to gRPC server")
	return nil
}

// Disconnect closes the connection to the gRPC server
func (c *GRPCClient) Disconnect() {
	if c.conn != nil {
		c.logger.Info("Disconnecting from gRPC server")
		c.conn.Close()
		c.conn = nil
		c.client = nil
	}
}

// RegisterDevice registers this device with the server
func (c *GRPCClient) RegisterDevice(device *pb.Device) (*pb.RegisterDeviceResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("client not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	req := &pb.RegisterDeviceRequest{
		Device: device,
	}

	c.logger.WithFields(logrus.Fields{
		"device_id":   device.Id,
		"device_name": device.Name,
		"device_type": device.Type,
	}).Info("Registering device")

	resp, err := c.client.RegisterDevice(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to register device: %w", err)
	}

	if resp.Success {
		c.logger.WithField("known_devices", len(resp.KnownDevices)).Info("Device registered successfully")
	} else {
		c.logger.WithField("error", resp.Message).Error("Device registration failed")
	}

	return resp, nil
}

// SendMessage sends a message to another device
func (c *GRPCClient) SendMessage(fromDeviceID, toDeviceID, messageType string, payload []byte, metadata map[string]string) (*pb.SendMessageResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("client not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	req := &pb.SendMessageRequest{
		FromDeviceId: fromDeviceID,
		ToDeviceId:   toDeviceID,
		MessageType:  messageType,
		Payload:      payload,
		Metadata:     metadata,
	}

	c.logger.WithFields(logrus.Fields{
		"from":         fromDeviceID,
		"to":           toDeviceID,
		"message_type": messageType,
		"payload_size": len(payload),
	}).Info("Sending message")

	resp, err := c.client.SendMessage(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to send message: %w", err)
	}

	if resp.Success {
		c.logger.Info("Message sent successfully")
	} else {
		c.logger.WithField("error", resp.Message).Error("Message sending failed")
	}

	return resp, nil
}

// SendHeartbeat sends a heartbeat to the server
func (c *GRPCClient) SendHeartbeat(deviceID string, capabilities *pb.DeviceCapabilities, status pb.DeviceStatus) (*pb.HeartbeatResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("client not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	req := &pb.HeartbeatRequest{
		DeviceId:     deviceID,
		Capabilities: capabilities,
		Status:       status,
	}

	c.logger.WithFields(logrus.Fields{
		"device_id": deviceID,
		"status":    status.String(),
	}).Debug("Sending heartbeat")

	resp, err := c.client.Heartbeat(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to send heartbeat: %w", err)
	}

	if !resp.Success {
		c.logger.Debug("Heartbeat failed, device may need to re-register")
	}

	return resp, nil
}

// GetDevices retrieves the list of available devices
func (c *GRPCClient) GetDevices(deviceID string) (*pb.GetDevicesResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("client not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	req := &pb.GetDevicesRequest{
		RequestingDeviceId: deviceID,
	}

	c.logger.WithField("device_id", deviceID).Debug("Getting devices list")

	resp, err := c.client.GetDevices(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get devices: %w", err)
	}

	c.logger.WithField("device_count", len(resp.Devices)).Debug("Retrieved devices list")

	return resp, nil
}

// SubmitTask submits a compute task for distribution
func (c *GRPCClient) SubmitTask(task *pb.ComputeTask, submittingDeviceID string, maxDevices, targetSubtasks int32) (*pb.SubmitTaskResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("client not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	req := &pb.SubmitTaskRequest{
		Task:               task,
		SubmittingDeviceId: submittingDeviceID,
		MaxDevices:         maxDevices,
		TargetSubtasks:     targetSubtasks,
	}

	c.logger.WithFields(logrus.Fields{
		"task_id":         task.TaskId,
		"task_type":       task.TaskType,
		"max_devices":     maxDevices,
		"target_subtasks": targetSubtasks,
	}).Info("Submitting compute task")

	resp, err := c.client.SubmitTask(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to submit task: %w", err)
	}

	if resp.Success {
		c.logger.WithField("task_id", resp.TaskId).Info("Task submitted successfully")
	} else {
		c.logger.WithField("error", resp.Message).Error("Task submission failed")
	}

	return resp, nil
}

// GetTaskStatus retrieves the status of a submitted task
func (c *GRPCClient) GetTaskStatus(taskID, requestingDeviceID string) (*pb.GetTaskStatusResponse, error) {
	if c.client == nil {
		return nil, fmt.Errorf("client not connected")
	}

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	req := &pb.GetTaskStatusRequest{
		TaskId:             taskID,
		RequestingDeviceId: requestingDeviceID,
	}

	c.logger.WithField("task_id", taskID).Debug("Getting task status")

	resp, err := c.client.GetTaskStatus(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get task status: %w", err)
	}

	return resp, nil
}

// SendTextMessage is a convenience method for sending text messages
func (c *GRPCClient) SendTextMessage(fromDeviceID, toDeviceID, message string) error {
	metadata := map[string]string{
		"content_type": "text/plain",
		"timestamp":    fmt.Sprintf("%d", time.Now().Unix()),
	}

	_, err := c.SendMessage(fromDeviceID, toDeviceID, "text", []byte(message), metadata)
	return err
}

// IsConnected returns true if the client is connected to the server
func (c *GRPCClient) IsConnected() bool {
	return c.conn != nil && c.client != nil
}
