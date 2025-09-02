#!/usr/bin/env python3
"""
Test UDP Discovery Between Macs
Quick diagnostic tool to check if UDP broadcasting works
"""

import socket
import json
import time
import threading
import uuid

def test_udp_broadcast():
    """Test if we can send UDP broadcasts"""
    print("🔍 Testing UDP broadcast capability...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        message = {
            "type": "test",
            "node_id": f"test_{uuid.uuid4().hex[:8]}",
            "timestamp": time.time(),
            "message": "UDP test from discompute"
        }
        
        data = json.dumps(message).encode('utf-8')
        sock.sendto(data, ('255.255.255.255', 5005))
        
        print("✅ UDP broadcast sent successfully")
        print(f"   Message: {message}")
        
    except Exception as e:
        print(f"❌ UDP broadcast failed: {e}")
        return False
    
    return True

def test_udp_listen():
    """Test if we can listen for UDP messages"""
    print("🔍 Testing UDP listen capability...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', 5005))
        sock.settimeout(5.0)  # 5 second timeout
        
        print("✅ UDP listen socket created successfully")
        print("   Listening on port 5005 for 5 seconds...")
        
        start_time = time.time()
        messages_received = 0
        
        while time.time() - start_time < 5:
            try:
                data, addr = sock.recvfrom(1024)
                message = json.loads(data.decode('utf-8'))
                print(f"   📥 Received from {addr}: {message.get('type', 'unknown')}")
                messages_received += 1
            except socket.timeout:
                continue
            except Exception as e:
                print(f"   ⚠️  Error receiving: {e}")
        
        print(f"✅ Received {messages_received} messages in 5 seconds")
        return True
        
    except Exception as e:
        print(f"❌ UDP listen failed: {e}")
        return False

def get_network_info():
    """Get network interface information"""
    print("🔍 Network interface information:")
    
    try:
        import netifaces
        
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    if ip and not ip.startswith('127.'):
                        print(f"   📡 {interface}: {ip}")
                        
    except ImportError:
        print("   ⚠️  netifaces not available, using basic method")
        
        # Fallback method
        hostname = socket.gethostname()
        try:
            ip = socket.gethostbyname(hostname)
            print(f"   📡 Primary IP: {ip}")
        except:
            print("   ❌ Could not determine IP address")

def test_firewall():
    """Test if firewall might be blocking"""
    print("🔍 Testing firewall and network connectivity...")
    
    # Test if we can connect to common ports
    test_ports = [80, 443, 22, 5005]
    
    for port in test_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:
                print(f"   ✅ Port {port}: Open")
            else:
                print(f"   ❌ Port {port}: Blocked/Closed")
                
        except Exception as e:
            print(f"   ⚠️  Port {port}: Error testing - {e}")

def main():
    print("🚀 Discompute UDP Discovery Test")
    print("================================")
    
    get_network_info()
    print()
    
    test_firewall()
    print()
    
    if test_udp_broadcast():
        print()
        test_udp_listen()
    
    print()
    print("🔧 Troubleshooting Tips:")
    print("======================")
    print("If UDP is blocked:")
    print("1. 📞 Contact IT to allow UDP port 5005")
    print("2. 🔗 Use manual discovery mode (specify IP addresses)")
    print("3. 🌐 Use a different discovery method (TCP, SSH, etc.)")
    print("4. 🏠 Test on home network first")
    print()
    print("Alternative solutions:")
    print("• Manual mode: --discovery-mode=manual --peers=192.168.1.100,192.168.1.101")
    print("• TCP discovery: Use TCP instead of UDP")
    print("• SSH tunneling: Create secure tunnels between Macs")

if __name__ == "__main__":
    main()
