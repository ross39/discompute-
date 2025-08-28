package device

import (
	"fmt"
	"sync"
	"time"

	pb "github.com/rossheaney/discompute/proto"
	"github.com/sirupsen/logrus"
)

// Registry manages the collection of discovered devices in a peer-to-peer manner
type Registry struct {
	mu      sync.RWMutex
	devices map[string]*pb.Device
	logger  *logrus.Logger
}

// NewRegistry creates a new device registry
func NewRegistry(logger *logrus.Logger) *Registry {
	return &Registry{
		devices: make(map[string]*pb.Device),
		logger:  logger,
	}
}

// RegisterDevice adds or updates a device in the registry
func (r *Registry) RegisterDevice(device *pb.Device) {
	r.mu.Lock()
	defer r.mu.Unlock()

	device.LastSeen = time.Now().Unix()
	r.devices[device.Id] = device

	r.logger.WithFields(logrus.Fields{
		"device_id":   device.Id,
		"device_name": device.Name,
		"device_type": device.Type,
		"address":     device.Address,
	}).Debug("Device registered")
}

// RegisterDeviceFromDiscovery creates and registers a device from discovery info
func (r *Registry) RegisterDeviceFromDiscovery(deviceInfo interface{}) {
	// Handle different discovery info types
	var device *pb.Device

	// Convert discovery info to protobuf device
	switch info := deviceInfo.(type) {
	case DiscoveredDevice: // From UDP discovery
		device = r.discoveredDeviceToProto(info)
	default:
		r.logger.Warn("Unknown device info type")
		return
	}

	r.RegisterDevice(device)
}

func (r *Registry) discoveredDeviceToProto(info DiscoveredDevice) *pb.Device {
	return &pb.Device{
		Id:      info.ID,
		Name:    info.Name,
		Type:    info.Type,
		Address: fmt.Sprintf("%s:%d", info.Address, info.Port),
		Capabilities: &pb.DeviceCapabilities{
			CpuCores:     0, // Will be populated from capabilities
			MemoryMb:     info.Capabilities.Memory,
			HasGpu:       info.Capabilities.Flops.FP16 > 0, // Assume GPU if has FP16
			GpuType:      info.Capabilities.Chip,
			BatteryLevel: -1, // Unknown for now
			IsCharging:   false,
			Fp32Tflops:   info.Capabilities.Flops.FP32,
			Fp16Tflops:   info.Capabilities.Flops.FP16,
			Int8Tflops:   info.Capabilities.Flops.INT8,
			Chip:         info.Capabilities.Chip,
		},
		LastSeen: info.LastSeen.Unix(),
		Status:   pb.DeviceStatus_AVAILABLE,
	}
}

// DiscoveredDevice represents discovered device information (from UDP discovery)
type DiscoveredDevice struct {
	ID           string
	Name         string
	Type         string
	Address      string
	Port         int
	Capabilities DeviceCapabilities
	LastSeen     time.Time
	Priority     int
}

// UnregisterDevice removes a device from the registry
func (r *Registry) UnregisterDevice(deviceID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.devices[deviceID]; exists {
		delete(r.devices, deviceID)
		r.logger.WithField("device_id", deviceID).Info("Device unregistered")
	}
}

// GetDevice returns a specific device by ID
func (r *Registry) GetDevice(deviceID string) (*pb.Device, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	device, exists := r.devices[deviceID]
	return device, exists
}

// GetAllDevices returns all registered devices
func (r *Registry) GetAllDevices() []*pb.Device {
	r.mu.RLock()
	defer r.mu.RUnlock()

	devices := make([]*pb.Device, 0, len(r.devices))
	for _, device := range r.devices {
		devices = append(devices, device)
	}

	return devices
}

// GetAvailableDevices returns devices that are currently available
func (r *Registry) GetAvailableDevices() []*pb.Device {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var available []*pb.Device
	for _, device := range r.devices {
		if device.Status == pb.DeviceStatus_AVAILABLE {
			available = append(available, device)
		}
	}

	return available
}

// UpdateDeviceStatus updates the status of a device
func (r *Registry) UpdateDeviceStatus(deviceID string, status pb.DeviceStatus) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	device, exists := r.devices[deviceID]
	if !exists {
		return false
	}

	device.Status = status
	device.LastSeen = time.Now().Unix()

	r.logger.WithFields(logrus.Fields{
		"device_id": deviceID,
		"status":    status.String(),
	}).Debug("Device status updated")

	return true
}

// UpdateDeviceCapabilities updates the capabilities of a device
func (r *Registry) UpdateDeviceCapabilities(deviceID string, capabilities *pb.DeviceCapabilities) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	device, exists := r.devices[deviceID]
	if !exists {
		return false
	}

	device.Capabilities = capabilities
	device.LastSeen = time.Now().Unix()

	return true
}

// CleanupStaleDevices removes devices that haven't been seen recently
func (r *Registry) CleanupStaleDevices(staleThreshold time.Duration) []string {
	r.mu.Lock()
	defer r.mu.Unlock()

	cutoff := time.Now().Add(-staleThreshold).Unix()
	var removedDevices []string

	for deviceID, device := range r.devices {
		if device.LastSeen < cutoff {
			delete(r.devices, deviceID)
			removedDevices = append(removedDevices, deviceID)

			r.logger.WithFields(logrus.Fields{
				"device_id": deviceID,
				"last_seen": time.Unix(device.LastSeen, 0),
			}).Info("Removed stale device")
		}
	}

	return removedDevices
}

// GetDeviceCount returns the total number of registered devices
func (r *Registry) GetDeviceCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.devices)
}

// GetDevicesByType returns devices of a specific type
func (r *Registry) GetDevicesByType(deviceType string) []*pb.Device {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var filtered []*pb.Device
	for _, device := range r.devices {
		if device.Type == deviceType {
			filtered = append(filtered, device)
		}
	}

	return filtered
}

// GetMobileDevices returns iOS and Android devices
func (r *Registry) GetMobileDevices() []*pb.Device {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var mobile []*pb.Device
	for _, device := range r.devices {
		if device.Type == "iphone" || device.Type == "ipad" || device.Type == "android" {
			mobile = append(mobile, device)
		}
	}

	return mobile
}

// GetComputeCapableDevices returns devices sorted by compute capability
func (r *Registry) GetComputeCapableDevices() []*pb.Device {
	devices := r.GetAvailableDevices()

	// Sort by approximate compute capability (simplified)
	// In a real implementation, this would use the FLOPS calculations
	return devices
}
