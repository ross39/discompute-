#!/usr/bin/env python3
"""
Discompute Mobile Client for iOS
Compatible with Pyto, a-Shell, and other Python iOS apps

This Python script implements the same UDP discovery protocol
as the Go version, allowing iOS devices to participate in the
discompute network.
"""

import socket
import json
import time
import threading
import platform
import subprocess
import sys
from datetime import datetime

class DiscomputeMobile:
    def __init__(self, device_id=None, port=8080):
        self.device_id = device_id or f"ios-{int(time.time())}"
        self.port = port
        self.listen_port = 5005
        self.broadcast_port = 5005
        self.running = False
        self.discovered_devices = {}
        self.device_capabilities = self.get_device_capabilities()
        
        print(f"🚀 Discompute Mobile Client")
        print(f"Device ID: {self.device_id}")
        print(f"Device Info: {self.device_capabilities}")
    
    def get_device_capabilities(self):
        """Detect iOS device capabilities"""
        try:
            # Try to get iOS device info
            model = platform.machine()
            if model.startswith('iPhone'):
                device_type = "iphone"
                # iPhone 16 Pro would be iPhone16,2 or similar
                if "16" in model:
                    chip = "Apple A18 Pro"
                    fp32_tflops = 2.80
                    fp16_tflops = 5.60
                    int8_tflops = 11.20
                elif "15" in model:
                    chip = "Apple A17 Pro" 
                    fp32_tflops = 2.15
                    fp16_tflops = 4.30
                    int8_tflops = 8.60
                else:
                    chip = "Apple A15 Bionic"
                    fp32_tflops = 1.37
                    fp16_tflops = 2.74
                    int8_tflops = 5.48
            elif model.startswith('iPad'):
                device_type = "ipad"
                chip = "Apple M2"  # Modern iPads use M-series
                fp32_tflops = 3.55
                fp16_tflops = 7.10
                int8_tflops = 14.20
            else:
                device_type = "ios_device"
                chip = "Unknown iOS Chip"
                fp32_tflops = 1.0
                fp16_tflops = 2.0
                int8_tflops = 4.0
                
            return {
                "model": model,
                "chip": chip,
                "type": device_type,
                "memory": 8192,  # Assume 8GB for modern iOS devices
                "flops": {
                    "fp32": fp32_tflops,
                    "fp16": fp16_tflops,
                    "int8": int8_tflops
                }
            }
        except Exception as e:
            print(f"Error detecting device capabilities: {e}")
            return {
                "model": "iOS Device",
                "chip": "Unknown",
                "type": "ios_device", 
                "memory": 4096,
                "flops": {"fp32": 1.0, "fp16": 2.0, "int8": 4.0}
            }
    
    def start_discovery(self):
        """Start UDP discovery service"""
        self.running = True
        
        # Start listening for discoveries
        listen_thread = threading.Thread(target=self.listen_for_devices, daemon=True)
        listen_thread.start()
        
        # Start broadcasting presence
        broadcast_thread = threading.Thread(target=self.broadcast_presence, daemon=True)  
        broadcast_thread.start()
        
        print(f"✅ Discovery started on port {self.listen_port}")
        return True
    
    def listen_for_devices(self):
        """Listen for UDP discovery messages"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.listen_port))
            sock.settimeout(1.0)  # Non-blocking with timeout
            
            print(f"👂 Listening for devices on port {self.listen_port}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(4096)
                    self.handle_discovery_message(data, addr[0])
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error receiving: {e}")
                        
        except Exception as e:
            print(f"Error setting up listener: {e}")
    
    def handle_discovery_message(self, data, sender_ip):
        """Handle received discovery message"""
        try:
            message = json.loads(data.decode('utf-8'))
            
            if message.get('type') == 'discovery' and message.get('node_id') != self.device_id:
                device_id = message['node_id']
                device_info = {
                    'id': device_id,
                    'name': message['device_capabilities']['model'],
                    'type': message['device_capabilities']['type'],
                    'address': sender_ip,
                    'port': message['grpc_port'],
                    'capabilities': message['device_capabilities'],
                    'last_seen': datetime.now(),
                    'priority': message.get('priority', 10)
                }
                
                is_new = device_id not in self.discovered_devices
                self.discovered_devices[device_id] = device_info
                
                if is_new:
                    print(f"🔍 Discovered: {device_info['name']} ({device_info['type']}) at {sender_ip}")
                    
        except Exception as e:
            print(f"Error handling discovery message: {e}")
    
    def broadcast_presence(self):
        """Broadcast our presence every 2.5 seconds"""
        while self.running:
            try:
                message = {
                    "type": "discovery",
                    "node_id": self.device_id,
                    "grpc_port": self.port,
                    "device_capabilities": self.device_capabilities,
                    "priority": 1,
                    "interface_name": "wifi",
                    "interface_type": "wifi",
                    "timestamp": int(time.time())
                }
                
                data = json.dumps(message).encode('utf-8')
                
                # Broadcast to local network
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.sendto(data, ('255.255.255.255', self.broadcast_port))
                sock.close()
                
                print(f"📡 Broadcasted presence ({len(self.discovered_devices)} devices known)")
                
            except Exception as e:
                print(f"Error broadcasting: {e}")
            
            time.sleep(2.5)  # EXO's broadcast interval
    
    def stop_discovery(self):
        """Stop discovery service"""
        self.running = False
        print("🛑 Discovery stopped")
    
    def list_devices(self):
        """List all discovered devices"""
        if not self.discovered_devices:
            print("No devices discovered yet")
            return
            
        print(f"\n📱 Discovered {len(self.discovered_devices)} device(s):")
        print("=" * 50)
        
        for device_id, device in self.discovered_devices.items():
            print(f"Device: {device['name']}")
            print(f"  ID: {device_id}")
            print(f"  Type: {device['type']}")
            print(f"  Address: {device['address']}:{device['port']}")
            caps = device['capabilities']
            if 'flops' in caps:
                print(f"  Compute: {caps['flops']['fp32']:.2f} TFLOPS (FP32)")
            print(f"  Last seen: {device['last_seen'].strftime('%H:%M:%S')}")
            print()
    
    def send_test_message(self, target_device_id, message):
        """Send a test message to another device"""
        if target_device_id not in self.discovered_devices:
            print(f"❌ Device {target_device_id} not found")
            return False
            
        device = self.discovered_devices[target_device_id]
        print(f"📤 Sending message to {device['name']}: {message}")
        
        # TODO: Implement actual gRPC message sending
        print("✅ Message sent (placeholder - gRPC client not implemented yet)")
        return True
    
    def get_stats(self):
        """Get service statistics"""
        uptime = time.time() - (time.time() if not hasattr(self, 'start_time') else self.start_time)
        return {
            'running': self.running,
            'device_id': self.device_id,
            'discovered_devices': len(self.discovered_devices),
            'device_type': self.device_capabilities['type'],
            'chip': self.device_capabilities['chip']
        }

def main():
    """Main function for interactive use"""
    print("🍎 Discompute iOS Client")
    print("Compatible with Pyto, a-Shell, and other Python iOS apps")
    print()
    
    # Create client
    client = DiscomputeMobile()
    
    try:
        # Start discovery
        client.start_discovery()
        
        print("\nCommands:")
        print("  'list' - Show discovered devices")
        print("  'info' - Show this device info") 
        print("  'stats' - Show statistics")
        print("  'send <device_id> <message>' - Send test message")
        print("  'quit' - Exit")
        print()
        
        while True:
            try:
                cmd = input("discompute> ").strip().lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'list':
                    client.list_devices()
                elif cmd == 'info':
                    caps = client.device_capabilities
                    print(f"Device: {caps['model']} ({caps['type']})")
                    print(f"Chip: {caps['chip']}")
                    print(f"Memory: {caps['memory']} MB")
                    print(f"Compute: {caps['flops']['fp32']:.2f} TFLOPS")
                elif cmd == 'stats':
                    stats = client.get_stats()
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                elif cmd.startswith('send '):
                    parts = cmd.split(' ', 2)
                    if len(parts) >= 3:
                        client.send_test_message(parts[1], parts[2])
                    else:
                        print("Usage: send <device_id> <message>")
                elif cmd == 'help':
                    print("Available commands: list, info, stats, send, quit")
                elif cmd == '':
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
                
    finally:
        client.stop_discovery()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
