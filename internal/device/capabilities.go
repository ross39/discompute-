package device

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/sirupsen/logrus"
)

// DeviceFlops represents compute capabilities in TFLOPS
type DeviceFlops struct {
	FP32 float64 `json:"fp32"`
	FP16 float64 `json:"fp16"`
	INT8 float64 `json:"int8"`
}

func (d DeviceFlops) String() string {
	return fmt.Sprintf("fp32: %.2f TFLOPS, fp16: %.2f TFLOPS, int8: %.2f TFLOPS", d.FP32, d.FP16, d.INT8)
}

// DeviceCapabilities represents comprehensive device information
type DeviceCapabilities struct {
	Model  string      `json:"model"`
	Chip   string      `json:"chip"`
	Memory int64       `json:"memory"` // in MB
	Flops  DeviceFlops `json:"flops"`
	Type   string      `json:"type"` // "mac", "iphone", "ipad", "linux", "android", etc.
}

func (d DeviceCapabilities) String() string {
	return fmt.Sprintf("Model: %s, Chip: %s, Memory: %dMB, Flops: %s, Type: %s", d.Model, d.Chip, d.Memory, d.Flops, d.Type)
}

// TFLOPS constant for calculations
const TFLOPS = 1.00

// ChipFlops maps chip names to their compute capabilities
// Based on EXO's comprehensive chip database
var ChipFlops = map[string]DeviceFlops{
	// Apple M chips (Mac)
	"Apple M1":       {FP32: 2.29 * TFLOPS, FP16: 4.58 * TFLOPS, INT8: 9.16 * TFLOPS},
	"Apple M1 Pro":   {FP32: 5.30 * TFLOPS, FP16: 10.60 * TFLOPS, INT8: 21.20 * TFLOPS},
	"Apple M1 Max":   {FP32: 10.60 * TFLOPS, FP16: 21.20 * TFLOPS, INT8: 42.40 * TFLOPS},
	"Apple M1 Ultra": {FP32: 21.20 * TFLOPS, FP16: 42.40 * TFLOPS, INT8: 84.80 * TFLOPS},
	"Apple M2":       {FP32: 3.55 * TFLOPS, FP16: 7.10 * TFLOPS, INT8: 14.20 * TFLOPS},
	"Apple M2 Pro":   {FP32: 5.68 * TFLOPS, FP16: 11.36 * TFLOPS, INT8: 22.72 * TFLOPS},
	"Apple M2 Max":   {FP32: 13.49 * TFLOPS, FP16: 26.98 * TFLOPS, INT8: 53.96 * TFLOPS},
	"Apple M2 Ultra": {FP32: 26.98 * TFLOPS, FP16: 53.96 * TFLOPS, INT8: 107.92 * TFLOPS},
	"Apple M3":       {FP32: 3.55 * TFLOPS, FP16: 7.10 * TFLOPS, INT8: 14.20 * TFLOPS},
	"Apple M3 Pro":   {FP32: 4.97 * TFLOPS, FP16: 9.94 * TFLOPS, INT8: 19.88 * TFLOPS},
	"Apple M3 Max":   {FP32: 14.20 * TFLOPS, FP16: 28.40 * TFLOPS, INT8: 56.80 * TFLOPS},
	"Apple M3 Ultra": {FP32: 54.26 * TFLOPS, FP16: 108.52 * TFLOPS, INT8: 217.04 * TFLOPS},
	"Apple M4":       {FP32: 4.26 * TFLOPS, FP16: 8.52 * TFLOPS, INT8: 17.04 * TFLOPS},
	"Apple M4 Pro":   {FP32: 5.72 * TFLOPS, FP16: 11.44 * TFLOPS, INT8: 22.88 * TFLOPS},
	"Apple M4 Max":   {FP32: 18.03 * TFLOPS, FP16: 36.07 * TFLOPS, INT8: 72.14 * TFLOPS},

	// Apple A chips (iPhone/iPad)
	"Apple A13 Bionic": {FP32: 0.69 * TFLOPS, FP16: 1.38 * TFLOPS, INT8: 2.76 * TFLOPS},
	"Apple A14 Bionic": {FP32: 0.75 * TFLOPS, FP16: 1.50 * TFLOPS, INT8: 3.00 * TFLOPS},
	"Apple A15 Bionic": {FP32: 1.37 * TFLOPS, FP16: 2.74 * TFLOPS, INT8: 5.48 * TFLOPS},
	"Apple A16 Bionic": {FP32: 1.79 * TFLOPS, FP16: 3.58 * TFLOPS, INT8: 7.16 * TFLOPS},
	"Apple A17 Pro":    {FP32: 2.15 * TFLOPS, FP16: 4.30 * TFLOPS, INT8: 8.60 * TFLOPS},
	"Apple A18":        {FP32: 2.30 * TFLOPS, FP16: 4.60 * TFLOPS, INT8: 9.20 * TFLOPS},  // Estimated
	"Apple A18 Pro":    {FP32: 2.80 * TFLOPS, FP16: 5.60 * TFLOPS, INT8: 11.20 * TFLOPS}, // iPhone 16 Pro

	// Common NVIDIA GPUs
	"NVIDIA GEFORCE RTX 4090": {FP32: 82.58 * TFLOPS, FP16: 165.16 * TFLOPS, INT8: 330.32 * TFLOPS},
	"NVIDIA GEFORCE RTX 4080": {FP32: 48.74 * TFLOPS, FP16: 97.48 * TFLOPS, INT8: 194.96 * TFLOPS},
	"NVIDIA GEFORCE RTX 4070": {FP32: 29.0 * TFLOPS, FP16: 58.0 * TFLOPS, INT8: 116.0 * TFLOPS},
	"NVIDIA GEFORCE RTX 3080": {FP32: 29.8 * TFLOPS, FP16: 59.6 * TFLOPS, INT8: 119.2 * TFLOPS},
	"NVIDIA GEFORCE RTX 3070": {FP32: 20.3 * TFLOPS, FP16: 40.6 * TFLOPS, INT8: 81.2 * TFLOPS},
	"NVIDIA GEFORCE RTX 3060": {FP32: 13.0 * TFLOPS, FP16: 26.0 * TFLOPS, INT8: 52.0 * TFLOPS},
}

var UnknownDeviceCapabilities = DeviceCapabilities{
	Model:  "Unknown Device",
	Chip:   "Unknown Chip",
	Memory: 0,
	Flops:  DeviceFlops{FP32: 0, FP16: 0, INT8: 0},
	Type:   "unknown",
}

// GetDeviceCapabilities detects the current device's capabilities
func GetDeviceCapabilities(logger *logrus.Logger) DeviceCapabilities {
	switch runtime.GOOS {
	case "darwin":
		return getAppleDeviceCapabilities(logger)
	case "linux":
		return getLinuxDeviceCapabilities(logger)
	case "android":
		return getAndroidDeviceCapabilities(logger)
	default:
		return getGenericDeviceCapabilities(logger)
	}
}

func getAppleDeviceCapabilities(logger *logrus.Logger) DeviceCapabilities {
	model := getAppleModel()
	chip := getAppleChip()
	memory := getSystemMemory()
	deviceType := determineAppleDeviceType(model)

	flops, exists := ChipFlops[chip]
	if !exists {
		logger.WithField("chip", chip).Warn("Unknown Apple chip, using default capabilities")
		flops = DeviceFlops{FP32: 1.0, FP16: 2.0, INT8: 4.0} // Conservative default
	}

	return DeviceCapabilities{
		Model:  model,
		Chip:   chip,
		Memory: memory,
		Flops:  flops,
		Type:   deviceType,
	}
}

func getAppleModel() string {
	// Try to get model from system_profiler on macOS
	if runtime.GOOS == "darwin" {
		cmd := exec.Command("system_profiler", "SPHardwareDataType")
		output, err := cmd.Output()
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Model Name:") {
					parts := strings.Split(line, ":")
					if len(parts) > 1 {
						return strings.TrimSpace(parts[1])
					}
				} else if strings.Contains(line, "Model Identifier:") {
					parts := strings.Split(line, ":")
					if len(parts) > 1 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}

	// Try sysctl on macOS/iOS
	cmd := exec.Command("sysctl", "-n", "hw.model")
	output, err := cmd.Output()
	if err == nil {
		return strings.TrimSpace(string(output))
	}

	return "Unknown Apple Device"
}

func getAppleChip() string {
	// Try to get chip from system_profiler on macOS
	if runtime.GOOS == "darwin" {
		cmd := exec.Command("system_profiler", "SPHardwareDataType")
		output, err := cmd.Output()
		if err == nil {
			lines := strings.Split(string(output), "\n")
			for _, line := range lines {
				if strings.Contains(line, "Chip:") {
					parts := strings.Split(line, ":")
					if len(parts) > 1 {
						return strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}

	// Try sysctl for CPU brand
	cmd := exec.Command("sysctl", "-n", "machdep.cpu.brand_string")
	output, err := cmd.Output()
	if err == nil {
		brand := strings.TrimSpace(string(output))
		if strings.Contains(brand, "Apple") {
			return brand
		}
	}

	// For iOS devices, try to detect based on architecture and model
	if runtime.GOARCH == "arm64" {
		model := getAppleModel()
		return inferChipFromModel(model)
	}

	return "Unknown Apple Chip"
}

func inferChipFromModel(model string) string {
	model = strings.ToLower(model)

	// iPhone models to chip mapping
	if strings.Contains(model, "iphone16") {
		if strings.Contains(model, "pro") {
			return "Apple A18 Pro"
		}
		return "Apple A18"
	}
	if strings.Contains(model, "iphone15") {
		if strings.Contains(model, "pro") {
			return "Apple A17 Pro"
		}
		return "Apple A16 Bionic"
	}
	if strings.Contains(model, "iphone14") {
		if strings.Contains(model, "pro") {
			return "Apple A16 Bionic"
		}
		return "Apple A15 Bionic"
	}
	if strings.Contains(model, "iphone13") {
		return "Apple A15 Bionic"
	}
	if strings.Contains(model, "iphone12") {
		return "Apple A14 Bionic"
	}

	// iPad models - many use similar chips to iPhones
	if strings.Contains(model, "ipad") {
		if strings.Contains(model, "pro") && strings.Contains(model, "2024") {
			return "Apple M4"
		}
		if strings.Contains(model, "pro") && strings.Contains(model, "2023") {
			return "Apple M2"
		}
		if strings.Contains(model, "air") && strings.Contains(model, "2024") {
			return "Apple M2"
		}
		// Default to A15 for newer iPads
		return "Apple A15 Bionic"
	}

	// Mac models
	if strings.Contains(model, "mac") {
		if strings.Contains(model, "m4") {
			return "Apple M4"
		}
		if strings.Contains(model, "m3") {
			return "Apple M3"
		}
		if strings.Contains(model, "m2") {
			return "Apple M2"
		}
		if strings.Contains(model, "m1") {
			return "Apple M1"
		}
	}

	return "Unknown Apple Chip"
}

func determineAppleDeviceType(model string) string {
	model = strings.ToLower(model)

	if strings.Contains(model, "iphone") {
		return "iphone"
	}
	if strings.Contains(model, "ipad") {
		return "ipad"
	}
	if strings.Contains(model, "mac") || strings.Contains(model, "macbook") {
		return "mac"
	}

	// Default to mac for macOS
	if runtime.GOOS == "darwin" {
		return "mac"
	}

	return "apple_device"
}

func getLinuxDeviceCapabilities(logger *logrus.Logger) DeviceCapabilities {
	model := "Linux Device"
	chip := "Unknown Linux Chip"
	memory := getSystemMemory()

	// Try to get CPU info
	if cpuInfo, err := cpu.Info(); err == nil && len(cpuInfo) > 0 {
		chip = cpuInfo[0].ModelName
		model = fmt.Sprintf("Linux Device (%s)", chip)
	}

	// Check for NVIDIA GPU
	// TODO: Add GPU detection for Linux

	return DeviceCapabilities{
		Model:  model,
		Chip:   chip,
		Memory: memory,
		Flops:  DeviceFlops{FP32: 1.0, FP16: 2.0, INT8: 4.0}, // Conservative default
		Type:   "linux",
	}
}

func getAndroidDeviceCapabilities(logger *logrus.Logger) DeviceCapabilities {
	model := "Android Device"
	chip := "Unknown Android Chip"
	memory := getSystemMemory()

	// Try to get Android system properties
	// TODO: Add Android-specific detection

	return DeviceCapabilities{
		Model:  model,
		Chip:   chip,
		Memory: memory,
		Flops:  DeviceFlops{FP32: 0.5, FP16: 1.0, INT8: 2.0}, // Conservative default
		Type:   "android",
	}
}

func getGenericDeviceCapabilities(logger *logrus.Logger) DeviceCapabilities {
	memory := getSystemMemory()

	return DeviceCapabilities{
		Model:  fmt.Sprintf("%s Device", runtime.GOOS),
		Chip:   "Unknown Chip",
		Memory: memory,
		Flops:  DeviceFlops{FP32: 0.1, FP16: 0.2, INT8: 0.4}, // Very conservative
		Type:   runtime.GOOS,
	}
}

func getSystemMemory() int64 {
	v, err := mem.VirtualMemory()
	if err != nil {
		return 0
	}
	return int64(v.Total / (1024 * 1024)) // Convert to MB
}

// GetDeviceID generates a unique device identifier
func GetDeviceID() string {
	// Try to get a stable hardware identifier
	if id := getAppleDeviceID(); id != "" {
		return id
	}

	// Fallback to MAC address hash or generated ID
	// TODO: Implement more robust device ID generation
	return generateRandomID()
}

func getAppleDeviceID() string {
	// Try to get hardware UUID on macOS
	cmd := exec.Command("system_profiler", "SPHardwareDataType")
	output, err := cmd.Output()
	if err == nil {
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "Hardware UUID:") {
				parts := strings.Split(line, ":")
				if len(parts) > 1 {
					return strings.TrimSpace(parts[1])
				}
			}
		}
	}

	return ""
}

func generateRandomID() string {
	// Simple random ID generation - in production, use more robust method
	return fmt.Sprintf("device-%d", runtime.NumCPU()*1000+int(getSystemMemory()%1000))
}
