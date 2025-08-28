package server

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/rossheaney/discompute/internal/device"
	pb "github.com/rossheaney/discompute/proto"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
)

// GRPCServer implements the DeviceService gRPC interface
type GRPCServer struct {
	pb.UnimplementedDeviceServiceServer

	registry *device.Registry
	logger   *logrus.Logger
	server   *grpc.Server
	listener net.Listener

	// Configuration
	port      int
	enableTLS bool
	certFile  string
	keyFile   string

	// Message handling
	messageHandlers map[string]MessageHandler
}

// MessageHandler defines the interface for handling different message types
type MessageHandler interface {
	HandleMessage(ctx context.Context, fromDeviceID, toDeviceID string, payload []byte, metadata map[string]string) error
}

// NewGRPCServer creates a new gRPC server
func NewGRPCServer(registry *device.Registry, port int, logger *logrus.Logger) *GRPCServer {
	return &GRPCServer{
		registry:        registry,
		logger:          logger,
		port:            port,
		messageHandlers: make(map[string]MessageHandler),
	}
}

// SetTLS configures TLS for the server
func (s *GRPCServer) SetTLS(certFile, keyFile string) {
	s.enableTLS = true
	s.certFile = certFile
	s.keyFile = keyFile
}

// RegisterMessageHandler registers a handler for a specific message type
func (s *GRPCServer) RegisterMessageHandler(messageType string, handler MessageHandler) {
	s.messageHandlers[messageType] = handler
}

// Start starts the gRPC server
func (s *GRPCServer) Start() error {
	var opts []grpc.ServerOption

	// Configure TLS if enabled
	if s.enableTLS {
		creds, err := credentials.NewServerTLSFromFile(s.certFile, s.keyFile)
		if err != nil {
			return fmt.Errorf("failed to load TLS credentials: %w", err)
		}
		opts = append(opts, grpc.Creds(creds))
		s.logger.Info("TLS enabled for gRPC server")
	}

	// Create the gRPC server
	s.server = grpc.NewServer(opts...)
	pb.RegisterDeviceServiceServer(s.server, s)

	// Create listener
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", s.port, err)
	}
	s.listener = listener

	s.logger.WithFields(logrus.Fields{
		"port": s.port,
		"tls":  s.enableTLS,
	}).Info("Starting gRPC server")

	// Start serving in a goroutine
	go func() {
		if err := s.server.Serve(listener); err != nil {
			s.logger.WithError(err).Error("gRPC server error")
		}
	}()

	return nil
}

// Stop stops the gRPC server
func (s *GRPCServer) Stop() {
	if s.server != nil {
		s.logger.Info("Stopping gRPC server")
		s.server.GracefulStop()
	}
	if s.listener != nil {
		s.listener.Close()
	}
}

// RegisterDevice implements the DeviceService.RegisterDevice RPC
func (s *GRPCServer) RegisterDevice(ctx context.Context, req *pb.RegisterDeviceRequest) (*pb.RegisterDeviceResponse, error) {
	// Get client address for logging
	clientAddr := "unknown"
	if peer, ok := peer.FromContext(ctx); ok {
		clientAddr = peer.Addr.String()
	}

	s.logger.WithFields(logrus.Fields{
		"device_id":   req.Device.Id,
		"device_name": req.Device.Name,
		"device_type": req.Device.Type,
		"client_addr": clientAddr,
	}).Info("Device registration request")

	// Validate device information
	if req.Device.Id == "" {
		return &pb.RegisterDeviceResponse{
			Success: false,
			Message: "Device ID is required",
		}, nil
	}

	if req.Device.Name == "" {
		return &pb.RegisterDeviceResponse{
			Success: false,
			Message: "Device name is required",
		}, nil
	}

	// Set device status to available by default
	if req.Device.Status == pb.DeviceStatus_UNKNOWN {
		req.Device.Status = pb.DeviceStatus_AVAILABLE
	}

	// Register the device
	s.registry.RegisterDevice(req.Device)

	// Return list of known devices
	knownDevices := s.registry.GetAllDevices()

	return &pb.RegisterDeviceResponse{
		Success:      true,
		Message:      "Device registered successfully",
		KnownDevices: knownDevices,
	}, nil
}

// SendMessage implements the DeviceService.SendMessage RPC
func (s *GRPCServer) SendMessage(ctx context.Context, req *pb.SendMessageRequest) (*pb.SendMessageResponse, error) {
	s.logger.WithFields(logrus.Fields{
		"from_device":  req.FromDeviceId,
		"to_device":    req.ToDeviceId,
		"message_type": req.MessageType,
		"payload_size": len(req.Payload),
	}).Info("Message send request")

	// Validate request
	if req.FromDeviceId == "" {
		return &pb.SendMessageResponse{
			Success: false,
			Message: "From device ID is required",
		}, nil
	}

	if req.ToDeviceId == "" {
		return &pb.SendMessageResponse{
			Success: false,
			Message: "To device ID is required",
		}, nil
	}

	// Check if target device exists and is available
	targetDevice, exists := s.registry.GetDevice(req.ToDeviceId)
	if !exists {
		return &pb.SendMessageResponse{
			Success: false,
			Message: "Target device not found",
		}, nil
	}

	if targetDevice.Status != pb.DeviceStatus_AVAILABLE {
		return &pb.SendMessageResponse{
			Success: false,
			Message: "Target device is not available",
		}, nil
	}

	// Handle the message based on type
	if handler, exists := s.messageHandlers[req.MessageType]; exists {
		err := handler.HandleMessage(ctx, req.FromDeviceId, req.ToDeviceId, req.Payload, req.Metadata)
		if err != nil {
			s.logger.WithError(err).Error("Message handler error")
			return &pb.SendMessageResponse{
				Success: false,
				Message: fmt.Sprintf("Message handling failed: %v", err),
			}, nil
		}
	} else {
		s.logger.WithField("message_type", req.MessageType).Debug("No handler for message type")
	}

	return &pb.SendMessageResponse{
		Success:   true,
		Message:   "Message sent successfully",
		Timestamp: time.Now().Unix(),
	}, nil
}

// Heartbeat implements the DeviceService.Heartbeat RPC
func (s *GRPCServer) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	s.logger.WithFields(logrus.Fields{
		"device_id": req.DeviceId,
		"status":    req.Status.String(),
	}).Debug("Heartbeat request")

	// Update device status and capabilities
	if !s.registry.UpdateDeviceStatus(req.DeviceId, req.Status) {
		// Device not found, it might need to register first
		return &pb.HeartbeatResponse{
			Success: false,
		}, nil
	}

	if req.Capabilities != nil {
		s.registry.UpdateDeviceCapabilities(req.DeviceId, req.Capabilities)
	}

	// Return updated device list
	updatedDevices := s.registry.GetAllDevices()

	return &pb.HeartbeatResponse{
		Success:        true,
		UpdatedDevices: updatedDevices,
	}, nil
}

// GetDevices implements the DeviceService.GetDevices RPC
func (s *GRPCServer) GetDevices(ctx context.Context, req *pb.GetDevicesRequest) (*pb.GetDevicesResponse, error) {
	s.logger.WithField("requesting_device", req.RequestingDeviceId).Debug("Get devices request")

	devices := s.registry.GetAllDevices()

	// Filter by type if requested
	if req.FilterType != "" {
		filtered := make([]*pb.Device, 0)
		for _, device := range devices {
			if device.Type == req.FilterType {
				filtered = append(filtered, device)
			}
		}
		devices = filtered
	}

	return &pb.GetDevicesResponse{
		Devices: devices,
	}, nil
}

// SubmitTask implements the DeviceService.SubmitTask RPC
func (s *GRPCServer) SubmitTask(ctx context.Context, req *pb.SubmitTaskRequest) (*pb.SubmitTaskResponse, error) {
	s.logger.WithFields(logrus.Fields{
		"task_id":           req.Task.TaskId,
		"task_type":         req.Task.TaskType,
		"submitting_device": req.SubmittingDeviceId,
		"max_devices":       req.MaxDevices,
		"target_subtasks":   req.TargetSubtasks,
	}).Info("Task submission request")

	// TODO: Implement task distribution logic
	// For now, return a placeholder response

	return &pb.SubmitTaskResponse{
		Success: true,
		Message: "Task submitted successfully (placeholder)",
		TaskId:  req.Task.TaskId,
	}, nil
}

// GetTaskStatus implements the DeviceService.GetTaskStatus RPC
func (s *GRPCServer) GetTaskStatus(ctx context.Context, req *pb.GetTaskStatusRequest) (*pb.GetTaskStatusResponse, error) {
	s.logger.WithFields(logrus.Fields{
		"task_id":           req.TaskId,
		"requesting_device": req.RequestingDeviceId,
	}).Debug("Task status request")

	// TODO: Implement task status tracking
	return &pb.GetTaskStatusResponse{
		Success: false,
	}, nil
}

// ExecuteSubtask implements the DeviceService.ExecuteSubtask RPC
func (s *GRPCServer) ExecuteSubtask(ctx context.Context, req *pb.ExecuteSubtaskRequest) (*pb.ExecuteSubtaskResponse, error) {
	s.logger.WithFields(logrus.Fields{
		"subtask_id":       req.Subtask.SubtaskId,
		"parent_task_id":   req.Subtask.ParentTaskId,
		"executing_device": req.ExecutingDeviceId,
	}).Info("Subtask execution request")

	// TODO: Implement subtask execution
	return &pb.ExecuteSubtaskResponse{
		Success: false,
		Message: "Subtask execution not yet implemented",
	}, nil
}

// GetPort returns the port the server is listening on
func (s *GRPCServer) GetPort() int {
	return s.port
}

// BasicMessageHandler implements a basic message handler for testing
type BasicMessageHandler struct {
	logger *logrus.Logger
}

// NewBasicMessageHandler creates a new basic message handler
func NewBasicMessageHandler(logger *logrus.Logger) *BasicMessageHandler {
	return &BasicMessageHandler{logger: logger}
}

// HandleMessage handles basic text messages
func (h *BasicMessageHandler) HandleMessage(ctx context.Context, fromDeviceID, toDeviceID string, payload []byte, metadata map[string]string) error {
	h.logger.WithFields(logrus.Fields{
		"from":     fromDeviceID,
		"to":       toDeviceID,
		"message":  string(payload),
		"metadata": metadata,
	}).Info("Handling basic message")

	// For now, just log the message
	// In the future, this could forward to the target device
	return nil
}
