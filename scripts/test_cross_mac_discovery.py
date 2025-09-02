#!/usr/bin/env python3
"""
Test Cross-Mac UDP Discovery
Run this on both Macs simultaneously to see if they can discover each other
"""

import socket
import json
import time
import threading
import uuid
import sys

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "unknown"

def broadcast_discovery(node_id, local_ip):
    """Broadcast discovery messages"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    print(f"📡 Broadcasting from {local_ip} as {node_id}")
    
    while True:
        try:
            message = {
                "type": "discovery",
                "node_id": node_id,
                "local_ip": local_ip,
                "grpc_port": 50051,
                "http_port": 8080,
                "timestamp": time.time(),
                "message": "Cross-Mac discovery test"
            }
            
            data = json.dumps(message).encode('utf-8')
            
            # Try multiple broadcast addresses
            broadcast_addresses = [
                '255.255.255.255',  # Global broadcast
                local_ip.rsplit('.', 1)[0] + '.255'  # Subnet broadcast
            ]
            
            for addr in broadcast_addresses:
                try:
                    sock.sendto(data, (addr, 5005))
                except Exception as e:
                    print(f"   ⚠️  Failed to broadcast to {addr}: {e}")
            
            time.sleep(3)  # Broadcast every 3 seconds
            
        except Exception as e:
            print(f"❌ Broadcast error: {e}")
            time.sleep(5)

def listen_for_discovery():
    """Listen for discovery messages from other Macs"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', 5005))
        
        print("👂 Listening for discovery messages on port 5005...")
        
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                message = json.loads(data.decode('utf-8'))
                
                if message.get('type') == 'discovery':
                    node_id = message.get('node_id', 'unknown')
                    remote_ip = message.get('local_ip', addr[0])
                    
                    print(f"🎯 DISCOVERED: {node_id} from {remote_ip} (received from {addr[0]})")
                    print(f"   Message: {message.get('message', 'no message')}")
                    print(f"   Timestamp: {time.ctime(message.get('timestamp', 0))}")
                    print()
                
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print(f"   ⚠️  Invalid JSON from {addr}")
            except Exception as e:
                print(f"   ⚠️  Error receiving from {addr}: {e}")
                
    except Exception as e:
        print(f"❌ Listen error: {e}")

def main():
    print("🚀 Cross-Mac UDP Discovery Test")
    print("===============================")
    
    local_ip = get_local_ip()
    node_id = f"test_mac_{uuid.uuid4().hex[:8]}"
    
    print(f"🖥️  This Mac: {node_id}")
    print(f"📍 Local IP: {local_ip}")
    print()
    print("Instructions:")
    print("1. Run this script on BOTH Macs simultaneously")
    print("2. Leave it running for 30-60 seconds")
    print("3. You should see 'DISCOVERED' messages if UDP works between Macs")
    print("4. Press Ctrl+C to stop")
    print()
    
    # Start listener in background thread
    listener_thread = threading.Thread(target=listen_for_discovery, daemon=True)
    listener_thread.start()
    
    # Wait a moment for listener to start
    time.sleep(1)
    
    try:
        # Start broadcasting
        broadcast_discovery(node_id, local_ip)
    except KeyboardInterrupt:
        print("\n🛑 Test stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
