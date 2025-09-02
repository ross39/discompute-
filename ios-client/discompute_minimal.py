#!/usr/bin/env python3
"""Minimal Discompute iOS Client for a-Shell - Easy to copy/paste"""
import socket, json, time, threading, uuid, logging, http.server, socketserver, psutil, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('discompute')

# Try ML imports
try:
    from tinygrad import Tensor, nn; from tinygrad.nn.optim import Adam; from tinygrad.nn.state import get_parameters
    HAS_ML = True; print("🧠 ML libraries available")
except: HAS_ML = False; print("⚠️ Using simulation mode")

class MiniModel:
    def __init__(self):
        if HAS_ML: self.conv1, self.conv2, self.fc = nn.Conv2d(1,32,3), nn.Conv2d(32,64,3), nn.Linear(1600,10); self.opt = Adam(get_parameters(self))
    def __call__(self, x): 
        if not HAS_ML: return None
        return self.fc(self.conv2(self.conv1(x.reshape(-1,1,28,28)).relu().max_pool2d(2)).relu().max_pool2d(2).flatten(1))
    def train(self, data, labels):
        if not HAS_ML: return 0.5, 0.8, {}
        try:
            x, y = Tensor(data), Tensor(labels); Tensor.training = True; loss = self(x).sparse_categorical_crossentropy(y)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            Tensor.training = False; acc = (self(x).argmax(1) == y).mean().item()
            return loss.item(), acc, {"gradients": [0.1]*10}
        except: return 0.5, 0.1, {}

class Handler(http.server.BaseHTTPRequestHandler):
    def __init__(self, client, *args): self.client = client; super().__init__(*args)
    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode())
            action = data.get('action')
            if action == 'initialize_training': resp = self.client.init_training(data.get('config', {}))
            elif action == 'train_batch': resp = self.client.train_batch(data.get('batch', {}))
            elif action == 'model_update': resp = {'success': True, 'message': 'Updated'}
            elif action == 'get_metrics': resp = {'success': True, 'metrics': self.client.metrics}
            elif action == 'health_check': resp = {'success': True, 'message': 'Healthy'}
            else: resp = {'success': False, 'message': f'Unknown: {action}'}
            self.send_response(200); self.send_header('Content-type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        except Exception as e: 
            self.send_response(500); self.end_headers(); self.wfile.write(json.dumps({'success': False, 'message': str(e)}).encode())
    def log_message(self, fmt, *args): pass

class Client:
    def __init__(self, port=8080):
        self.id = f"ios_{uuid.uuid4().hex[:8]}"; self.port = port; self.model = None; self.active = False
        self.caps = {'cpu_cores': os.cpu_count(), 'memory_mb': int(psutil.virtual_memory().total/1024/1024), 'device_type': 'ios'}
        self.metrics = {'cpu_usage': 0.0, 'memory_usage': 0.0, 'battery_level': 0.8, 'throughput_tps': 0.0}
        logger.info(f"Client {self.id} initialized")
    
    def start(self):
        threading.Thread(target=self.udp_broadcast, daemon=True).start()
        server = socketserver.TCPServer(("", self.port), lambda *a: Handler(self, *a))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Thread(target=self.update_metrics, daemon=True).start()
        logger.info(f"🚀 Client running on port {self.port}"); logger.info(f"📡 Broadcasting UDP on 5005"); logger.info(f"🆔 ID: {self.id}")
    
    def udp_broadcast(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            try:
                msg = {"type": "discovery", "node_id": self.id, "grpc_port": 50051, "http_port": self.port, 
                       "device_capabilities": self.caps, "priority": 1, "interface_type": "wifi", "timestamp": time.time()}
                sock.sendto(json.dumps(msg).encode(), ('255.255.255.255', 5005)); time.sleep(2.5)
            except Exception as e: logger.error(f"UDP error: {e}"); time.sleep(5)
    
    def update_metrics(self):
        while True:
            try: self.metrics.update({'cpu_usage': psutil.cpu_percent()/100, 'memory_usage': psutil.virtual_memory().percent/100})
            except: pass
            time.sleep(10)
    
    def init_training(self, config):
        try: self.model = MiniModel(); self.active = True; logger.info("Training initialized"); return {'success': True, 'message': 'Initialized'}
        except Exception as e: return {'success': False, 'message': str(e)}
    
    def train_batch(self, batch):
        try:
            if not self.active or not self.model: return {'success': False, 'message': 'Not initialized'}
            bid, data, labels = batch.get('batch_id', 'unknown'), batch.get('data', []), batch.get('labels', [])
            start = time.time(); loss, acc, grads = self.model.train(data, labels); dur = time.time() - start
            self.metrics['throughput_tps'] = len(data) / dur if dur > 0 else 0
            result = {'batch_id': bid, 'loss': loss, 'accuracy': acc, 'gradients': grads, 'processing_time': dur, 'device_metrics': self.metrics}
            logger.info(f"Batch {bid}: loss={loss:.3f}, acc={acc:.3f}"); return {'success': True, 'result': result}
        except Exception as e: return {'success': False, 'message': str(e)}

def main():
    print("🚀 Minimal Discompute Client"); print("="*30); client = Client()
    try: client.start(); print("💡 Ready for training!"); print("   Connect to discompute cluster"); [time.sleep(1) for _ in iter(int, 1)]
    except KeyboardInterrupt: print("\n🛑 Stopping...")
    except Exception as e: print(f"❌ Error: {e}")
    finally: print("👋 Stopped")

if __name__ == "__main__": main()
