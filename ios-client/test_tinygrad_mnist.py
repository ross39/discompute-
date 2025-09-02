#!/usr/bin/env python3
"""
Test Tinygrad MNIST implementation
This will help us debug and get a working neural network
"""

try:
    from tinygrad import Tensor, nn, dtypes, TinyJit
    from tinygrad.nn.optim import SGD, Adam
    from tinygrad.nn.datasets import mnist
    from tinygrad.nn.state import get_parameters
    import numpy as np
    
    print("✅ Tinygrad imports successful")
except ImportError as e:
    print(f"❌ Tinygrad import failed: {e}")
    exit(1)

def create_mnist_model():
    """Create MNIST model following official Tinygrad docs"""
    print("🧠 Creating MNIST model (following official docs)...")
    
    # Model from official Tinygrad MNIST tutorial
    class Model:
        def __init__(self):
            self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
            self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
            self.l3 = nn.Linear(1600, 10)

        def __call__(self, x: Tensor) -> Tensor:
            x = self.l1(x).relu().max_pool2d((2,2))
            x = self.l2(x).relu().max_pool2d((2,2))
            return self.l3(x.flatten(1).dropout(0.5))
    
    model = Model()
    
    # Use get_parameters as shown in docs
    optimizer = Adam(get_parameters(model))
    
    print("✅ Official MNIST model created")
    return model, optimizer

def load_mnist_data():
    """Load MNIST data using official Tinygrad function"""
    print("📊 Loading MNIST data...")
    
    try:
        X_train, Y_train, X_test, Y_test = mnist()
        print(f"✅ MNIST loaded: train={X_train.shape}, test={X_test.shape}")
        print(f"   Data types: X={X_train.dtype}, Y={Y_train.dtype}")
        return X_train, Y_train, X_test, Y_test
    except Exception as e:
        print(f"❌ Failed to load MNIST: {e}")
        return None, None, None, None

def test_training():
    """Test training following official Tinygrad MNIST tutorial"""
    print("🏃‍♂️ Testing training (official tutorial method)...")
    
    # Load real MNIST data
    X_train, Y_train, X_test, Y_test = load_mnist_data()
    if X_train is None:
        print("❌ Cannot proceed without MNIST data")
        return
    
    # Create model
    model, optimizer = create_mnist_model()
    
    print(f"📊 Data shapes: train={X_train.shape}, test={X_test.shape}")
    
    # Test model before training
    print("🧪 Testing untrained model...")
    Tensor.training = False
    acc = (model(X_test).argmax(axis=1) == Y_test).mean()
    print(f"   Untrained accuracy: {acc.item()*100:.2f}% (should be ~10%)")
    
    # Define training step function (as per docs)
    batch_size = 32  # Smaller batch for testing
    
    def step():
        Tensor.training = True  # Enable dropout
        # Random sampling as per docs
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        X, Y = X_train[samples], Y_train[samples]
        
        optimizer.zero_grad()
        loss = model(X).sparse_categorical_crossentropy(Y)
        loss.backward()
        optimizer.step()
        return loss
    
    # Create JIT version for speed (as per docs)
    print("🚀 Creating JIT compiled training step...")
    jit_step = TinyJit(step)
    
    # Training loop (following official tutorial)
    print("🏋️‍♂️ Starting training...")
    for i in range(100):  # 100 steps for quick test
        loss = jit_step()
        
        if i % 20 == 0:
            # Evaluate
            Tensor.training = False
            acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
            print(f"   Step {i:3d}: loss={loss.item():.3f}, acc={acc*100:.2f}%")
    
    # Final evaluation
    Tensor.training = False
    final_acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"\n🎉 Final accuracy: {final_acc*100:.2f}%")
    
    if final_acc > 0.95:
        print("🏆 Excellent! >95% accuracy achieved!")
    elif final_acc > 0.90:
        print("👍 Good! >90% accuracy achieved!")
    else:
        print("📚 Learning in progress...")
    
    return final_acc

def main():
    print("🧪 Testing Tinygrad MNIST Implementation")
    print("=" * 50)
    
    try:
        test_training()
        print("\n🎉 All tests passed! Tinygrad is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
