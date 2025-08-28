# iOS Client for Non-Jailbroken Devices

This directory contains iOS deployment options for standard (non-jailbroken) iOS devices like your iPhone 16 Pro and iPad Pro.

## 🚀 Quick Start Options

### Option 1: Python App (Recommended for Testing)

Use **Pyto** or **a-Shell** from the App Store to run our Python client:

#### Step 1: Install Python App
- **Pyto** (App Store): Full Python 3 environment
- **a-Shell** (App Store): Unix shell with Python support

#### Step 2: Copy Python Script
1. Copy `discompute_mobile.py` to your iPhone/iPad
2. Open in Pyto or a-Shell
3. Run: `python3 discompute_mobile.py`

#### Step 3: Test Discovery
```python
# The script will start automatically
discompute> list        # Show discovered devices
discompute> info        # Show your device capabilities  
discompute> stats       # Show connection statistics
discompute> quit        # Exit
```

### Option 2: Shortcuts App Integration

Create an iOS Shortcut that runs the discovery protocol:

1. Open **Shortcuts** app
2. Create new shortcut: "Discompute Discovery"
3. Add "Run Shell Script" action (requires a-Shell)
4. Paste the Python script

### Option 3: Native iOS App (Advanced)

For production use, create a native iOS app:

#### Requirements:
- Xcode 15+
- iOS 17+ deployment target
- Apple Developer account (for device deployment)

#### Quick Setup:
```bash
# Create iOS app project
cd ios-client
./create-ios-app.sh
```

## 📱 Device Detection

The Python client automatically detects your device:

### iPhone 16 Pro Detection:
- **Model**: iPhone16,2 (or similar)
- **Chip**: Apple A18 Pro  
- **Compute**: 2.80 TFLOPS (FP32)
- **Type**: iphone

### iPad Pro Detection:
- **Model**: iPad14,x (varies by model)
- **Chip**: Apple M4 (latest models)
- **Compute**: 4.26+ TFLOPS (FP32)
- **Type**: ipad

## 🔄 Testing Workflow

### 1. Start Mac Service:
```bash
cd /Users/rossheaney/Code/discompute
./bin/discompute start --debug --port 8080
```

### 2. Start iOS Client:
On your iPhone/iPad in Pyto:
```python
python3 discompute_mobile.py
```

### 3. Verify Discovery:
Both devices should discover each other within ~5 seconds.

Mac output:
```
INFO[0000] Discovered new device: iPhone16,2 (iphone) at 192.168.1.100
```

iOS output:
```
🔍 Discovered: MacBook Pro (mac) at 192.168.1.101
```

### 4. Test Communication:
```bash
# On Mac
./bin/discompute list
./bin/discompute send [iphone-device-id] "Hello iPhone!"

# On iOS  
discompute> list
discompute> send [mac-device-id] "Hello Mac!"
```

## 🛠 Troubleshooting

### Common Issues:

1. **No Devices Found:**
   - Ensure both devices on same WiFi network
   - Check firewall settings
   - Try different ports (5005, 5006, etc.)

2. **Python App Permissions:**
   - Grant network permissions in iOS Settings
   - Allow background app refresh

3. **Discovery Timeout:**
   - Wait 10-15 seconds for initial discovery
   - Restart the Python script if needed

### Debug Mode:
Add debug output to the Python script:
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Testing

Test your iPhone 16 Pro's compute capabilities:

```python
# In the Python client
discompute> info
Device: iPhone16,2 (iphone)
Chip: Apple A18 Pro
Memory: 8192 MB
Compute: 2.80 TFLOPS
```

This gives you ~2.8 TFLOPS of compute power to contribute to the distributed cluster!

## 🎯 Next Steps

Once discovery is working:

1. **Implement gRPC client** in Python for full communication
2. **Add compute task execution** for actual distributed processing  
3. **Create native iOS app** for App Store distribution
4. **Optimize for battery life** and background processing

## 🔗 Files in this Directory

- `discompute_mobile.py` - Python client for Pyto/a-Shell
- `README.md` - This deployment guide
- `create-ios-app.sh` - Native iOS app generator (coming soon)
- `requirements.txt` - Python dependencies

Ready to test your iPhone 16 Pro with your Mac! 🚀
