package discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/rossheaney/discompute/internal/device"
	"github.com/sirupsen/logrus"
)

// DiscoveryMessage represents the UDP broadcast message
type DiscoveryMessage struct {
	Type               string                    `json:"type"`
	NodeID             string                    `json:"node_id"`
	GRPCPort           int                       `json:"grpc_port"`
	DeviceCapabilities device.DeviceCapabilities `json:"device_capabilities"`
	Priority           int                       `json:"priority"`
	InterfaceName      string                    `json:"interface_name"`
	InterfaceType      string                    `json:"interface_type"`
	Timestamp          int64                     `json:"timestamp"`
}

// DiscoveredDevice represents a discovered device
type DiscoveredDevice struct {
	ID           string                    `json:"id"`
	Name         string                    `json:"name"`
	Type         string                    `json:"type"`
	Address      string                    `json:"address"`
	Port         int                       `json:"port"`
	Capabilities device.DeviceCapabilities `json:"capabilities"`
	LastSeen     time.Time                 `json:"last_seen"`
	Priority     int                       `json:"priority"`
}

// UDPDiscovery handles device discovery using UDP broadcasts
type UDPDiscovery struct {
	logger   *logrus.Logger
	nodeID   string
	nodePort int

	// Network settings
	listenPort    int
	broadcastPort int
	broadcastAddr string

	// Device information
	deviceCapabilities device.DeviceCapabilities

	// Discovery callbacks
	onDeviceFound func(DiscoveredDevice)
	onDeviceLost  func(string) // device ID

	// Internal state
	mu                sync.RWMutex
	discoveredDevices map[string]DiscoveredDevice
	running           bool
	ctx               context.Context
	cancel            context.CancelFunc

	// Network connections
	listenConn    *net.UDPConn
	broadcastConn *net.UDPConn

	// Intervals
	broadcastInterval time.Duration
	cleanupInterval   time.Duration
	discoveryTimeout  time.Duration
}

// NewUDPDiscovery creates a new UDP discovery service
func NewUDPDiscovery(nodeID string, nodePort int, logger *logrus.Logger) *UDPDiscovery {
	return &UDPDiscovery{
		logger:            logger,
		nodeID:            nodeID,
		nodePort:          nodePort,
		listenPort:        5005, // Default listen port
		broadcastPort:     5005, // Default broadcast port
		broadcastAddr:     "255.255.255.255",
		discoveredDevices: make(map[string]DiscoveredDevice),
		broadcastInterval: 2500 * time.Millisecond, // Like EXO's 2.5s
		cleanupInterval:   30 * time.Second,
		discoveryTimeout:  30 * time.Second,
	}
}

// SetPorts configures the UDP ports
func (u *UDPDiscovery) SetPorts(listenPort, broadcastPort int) {
	u.listenPort = listenPort
	u.broadcastPort = broadcastPort
}

// SetCallbacks sets the discovery event callbacks
func (u *UDPDiscovery) SetCallbacks(onFound func(DiscoveredDevice), onLost func(string)) {
	u.onDeviceFound = onFound
	u.onDeviceLost = onLost
}

// Start begins the UDP discovery service
func (u *UDPDiscovery) Start(ctx context.Context) error {
	u.mu.Lock()
	defer u.mu.Unlock()

	if u.running {
		return fmt.Errorf("UDP discovery service already running")
	}

	u.ctx, u.cancel = context.WithCancel(ctx)

	// Get device capabilities
	u.deviceCapabilities = device.GetDeviceCapabilities(u.logger)

	u.logger.WithFields(logrus.Fields{
		"node_id":        u.nodeID,
		"listen_port":    u.listenPort,
		"broadcast_port": u.broadcastPort,
		"capabilities":   u.deviceCapabilities.String(),
	}).Info("Starting UDP discovery service")

	// Start listening for discovery messages
	if err := u.startListening(); err != nil {
		return fmt.Errorf("failed to start listening: %w", err)
	}

	// Start broadcasting presence
	go u.broadcastPresence()

	// Start cleanup routine
	go u.cleanupRoutine()

	u.running = true
	return nil
}

// Stop stops the UDP discovery service
func (u *UDPDiscovery) Stop() {
	u.mu.Lock()
	defer u.mu.Unlock()

	if !u.running {
		return
	}

	u.cancel()

	if u.listenConn != nil {
		u.listenConn.Close()
	}
	if u.broadcastConn != nil {
		u.broadcastConn.Close()
	}

	u.running = false
	u.logger.Info("UDP discovery service stopped")
}

// GetDiscoveredDevices returns a copy of currently discovered devices
func (u *UDPDiscovery) GetDiscoveredDevices() map[string]DiscoveredDevice {
	u.mu.RLock()
	defer u.mu.RUnlock()

	devices := make(map[string]DiscoveredDevice)
	for id, device := range u.discoveredDevices {
		devices[id] = device
	}
	return devices
}

func (u *UDPDiscovery) startListening() error {
	// Create UDP address for listening
	listenAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf(":%d", u.listenPort))
	if err != nil {
		return err
	}

	// Create listening connection
	conn, err := net.ListenUDP("udp", listenAddr)
	if err != nil {
		return err
	}

	u.listenConn = conn

	// Start listening goroutine
	go u.listenForMessages()

	return nil
}

func (u *UDPDiscovery) listenForMessages() {
	buffer := make([]byte, 4096)

	for {
		select {
		case <-u.ctx.Done():
			return
		default:
			// Set read timeout to allow checking context
			u.listenConn.SetReadDeadline(time.Now().Add(1 * time.Second))

			n, addr, err := u.listenConn.ReadFromUDP(buffer)
			if err != nil {
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					continue // Timeout is expected, check context and continue
				}
				if u.ctx.Err() == nil { // Only log if not shutting down
					u.logger.WithError(err).Debug("Error reading UDP message")
				}
				continue
			}

			u.handleDiscoveryMessage(buffer[:n], addr)
		}
	}
}

func (u *UDPDiscovery) handleDiscoveryMessage(data []byte, addr *net.UDPAddr) {
	var msg DiscoveryMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		u.logger.WithError(err).Debug("Failed to unmarshal discovery message")
		return
	}

	// Ignore our own messages
	if msg.NodeID == u.nodeID {
		return
	}

	// Only handle discovery messages
	if msg.Type != "discovery" {
		return
	}

	u.logger.WithFields(logrus.Fields{
		"from_node":    msg.NodeID,
		"from_address": addr.String(),
		"device_type":  msg.DeviceCapabilities.Type,
		"device_model": msg.DeviceCapabilities.Model,
	}).Debug("Received discovery message")

	// Create device info
	deviceInfo := DiscoveredDevice{
		ID:           msg.NodeID,
		Name:         msg.DeviceCapabilities.Model,
		Type:         msg.DeviceCapabilities.Type,
		Address:      addr.IP.String(),
		Port:         msg.GRPCPort,
		Capabilities: msg.DeviceCapabilities,
		LastSeen:     time.Now(),
		Priority:     msg.Priority,
	}

	u.mu.Lock()
	existing, exists := u.discoveredDevices[msg.NodeID]
	u.discoveredDevices[msg.NodeID] = deviceInfo
	u.mu.Unlock()

	if !exists {
		u.logger.WithFields(logrus.Fields{
			"device_id":   msg.NodeID,
			"device_name": deviceInfo.Name,
			"address":     fmt.Sprintf("%s:%d", deviceInfo.Address, deviceInfo.Port),
			"device_type": deviceInfo.Type,
		}).Info("Discovered new device")

		if u.onDeviceFound != nil {
			go u.onDeviceFound(deviceInfo)
		}
	} else if existing.Address != deviceInfo.Address || existing.Port != deviceInfo.Port {
		u.logger.WithFields(logrus.Fields{
			"device_id":   msg.NodeID,
			"old_address": fmt.Sprintf("%s:%d", existing.Address, existing.Port),
			"new_address": fmt.Sprintf("%s:%d", deviceInfo.Address, deviceInfo.Port),
		}).Info("Device address updated")

		if u.onDeviceFound != nil {
			go u.onDeviceFound(deviceInfo)
		}
	}
}

func (u *UDPDiscovery) broadcastPresence() {
	ticker := time.NewTicker(u.broadcastInterval)
	defer ticker.Stop()

	for {
		select {
		case <-u.ctx.Done():
			return
		case <-ticker.C:
			u.sendBroadcast()
		}
	}
}

func (u *UDPDiscovery) sendBroadcast() {
	// Get all network interfaces
	interfaces := u.getNetworkInterfaces()

	for _, iface := range interfaces {
		u.sendBroadcastOnInterface(iface)
	}
}

func (u *UDPDiscovery) sendBroadcastOnInterface(iface NetworkInterface) {
	// Create broadcast message
	msg := DiscoveryMessage{
		Type:               "discovery",
		NodeID:             u.nodeID,
		GRPCPort:           u.nodePort,
		DeviceCapabilities: u.deviceCapabilities,
		Priority:           iface.Priority,
		InterfaceName:      iface.Name,
		InterfaceType:      iface.Type,
		Timestamp:          time.Now().Unix(),
	}

	data, err := json.Marshal(msg)
	if err != nil {
		u.logger.WithError(err).Error("Failed to marshal discovery message")
		return
	}

	// Create UDP connection for broadcasting
	conn, err := net.Dial("udp", fmt.Sprintf("%s:%d", u.broadcastAddr, u.broadcastPort))
	if err != nil {
		u.logger.WithError(err).Debug("Failed to create broadcast connection")
		return
	}
	defer conn.Close()

	// Send broadcast
	_, err = conn.Write(data)
	if err != nil {
		u.logger.WithError(err).Debug("Failed to send broadcast message")
		return
	}

	u.logger.WithFields(logrus.Fields{
		"interface":    iface.Name,
		"broadcast_to": fmt.Sprintf("%s:%d", u.broadcastAddr, u.broadcastPort),
		"device_type":  u.deviceCapabilities.Type,
	}).Debug("Sent discovery broadcast")
}

type NetworkInterface struct {
	Name     string
	Type     string
	Priority int
	Address  string
}

func (u *UDPDiscovery) getNetworkInterfaces() []NetworkInterface {
	var interfaces []NetworkInterface

	// Get system network interfaces
	ifaces, err := net.Interfaces()
	if err != nil {
		u.logger.WithError(err).Error("Failed to get network interfaces")
		return interfaces
	}

	for _, iface := range ifaces {
		// Skip loopback and down interfaces
		if iface.Flags&net.FlagLoopback != 0 || iface.Flags&net.FlagUp == 0 {
			continue
		}

		// Get addresses for this interface
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}

		for _, addr := range addrs {
			var ip net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}

			// Only use IPv4 addresses
			if ip == nil || ip.IsLoopback() || ip.To4() == nil {
				continue
			}

			// Determine interface type and priority
			interfaceType, priority := u.getInterfaceTypeAndPriority(iface.Name)

			interfaces = append(interfaces, NetworkInterface{
				Name:     iface.Name,
				Type:     interfaceType,
				Priority: priority,
				Address:  ip.String(),
			})
			break // Only use first valid address per interface
		}
	}

	return interfaces
}

func (u *UDPDiscovery) getInterfaceTypeAndPriority(name string) (string, int) {
	name = fmt.Sprintf("%s", name) // Normalize name

	// Wi-Fi interfaces (highest priority for mobile devices)
	if containsAny(name, []string{"wlan", "wifi", "en0", "en1"}) {
		return "wifi", 1
	}

	// Ethernet interfaces
	if containsAny(name, []string{"eth", "en", "ethernet"}) {
		return "ethernet", 2
	}

	// Cellular interfaces (important for mobile)
	if containsAny(name, []string{"cellular", "pdp_ip", "rmnet"}) {
		return "cellular", 3
	}

	// USB/tethering
	if containsAny(name, []string{"usb", "rndis"}) {
		return "usb", 4
	}

	// Default
	return "unknown", 10
}

func containsAny(str string, substrings []string) bool {
	for _, substr := range substrings {
		if fmt.Sprintf("%s", str) == substr || fmt.Sprintf("%s", str)[:min(len(str), len(substr))] == substr {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (u *UDPDiscovery) cleanupRoutine() {
	ticker := time.NewTicker(u.cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-u.ctx.Done():
			return
		case <-ticker.C:
			u.cleanupStaleDevices()
		}
	}
}

func (u *UDPDiscovery) cleanupStaleDevices() {
	u.mu.Lock()
	defer u.mu.Unlock()

	staleThreshold := time.Now().Add(-u.discoveryTimeout)
	var removedDevices []string

	for deviceID, device := range u.discoveredDevices {
		if device.LastSeen.Before(staleThreshold) {
			delete(u.discoveredDevices, deviceID)
			removedDevices = append(removedDevices, deviceID)

			u.logger.WithFields(logrus.Fields{
				"device_id": deviceID,
				"last_seen": device.LastSeen,
			}).Info("Removed stale device")
		}
	}

	// Notify about lost devices
	for _, deviceID := range removedDevices {
		if u.onDeviceLost != nil {
			go u.onDeviceLost(deviceID)
		}
	}
}

// GetLocalIP returns the local IP address for the best interface
func GetLocalIP() (string, error) {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "", err
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP.String(), nil
}
