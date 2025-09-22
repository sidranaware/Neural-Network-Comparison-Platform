# Benchmark aggregation script

import os, glob, json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.datasets import cifar10
import larq as lq

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

RESULTS_ROOT = "results"
MODEL_FILES = {
    "CNN": "cnn_baseline_updated.h5",
    "BNN": "bnn_baseline_final1.h5", 
    "HYBRID": "bnn_plus_hybrid_best.h5"
}

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_and_preprocess_data(num_samples=1000):
    """Load and preprocess CIFAR-10 test data"""
    print(f"📥 Loading CIFAR-10 test data ({num_samples} samples)...")
    (_, _), (x_test, y_test) = cifar10.load_data()
    
    # Normalize
    x_test = x_test.astype('float32') / 255.0
    
    # Take subset for benchmarking
    if num_samples < len(x_test):
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        x_test = x_test[indices]
        y_test = y_test[indices]
    
    print(f"✅ Loaded {len(x_test)} test samples")
    return x_test, y_test

def benchmark_model(model_name, model_path, x_test, y_test):
    """Benchmark a single model"""
    print(f"\n🔄 Benchmarking {model_name}...")
    
    # Load model with proper Larq handling
    try:
        if model_name in ["BNN", "HYBRID"]:
            # Try different approaches for Larq models
            try:
                # First try with larq.custom_objects
                model = tf.keras.models.load_model(model_path, custom_objects=lq.custom_objects, compile=False)
            except AttributeError:
                # If custom_objects doesn't exist, try with quantizers
                try:
                    custom_objects = {
                        'QuantDense': lq.layers.QuantDense,
                        'QuantConv2D': lq.layers.QuantConv2D,
                        'DoReFaQuantizer': lq.quantizers.DoReFaQuantizer,
                        'SteSign': lq.quantizers.SteSign,
                        'ApproxSign': lq.quantizers.ApproxSign,
                        'MagnitudeAwareSign': lq.quantizers.MagnitudeAwareSign,
                    }
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                except:
                    # Last resort - try loading without custom objects
                    model = tf.keras.models.load_model(model_path, compile=False)
        else:
            model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Model {model_name} loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        print(f"   Trying alternative loading method...")
        try:
            # Alternative: Load with TensorFlow only
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Model {model_name} loaded with alternative method")
        except Exception as e2:
            print(f"❌ Alternative loading also failed: {e2}")
            return None
    
    # Model info
    total_params = model.count_params()
    model_size_mb = os.path.getsize(model_path) / (1024*1024)
    
    print(f"📊 Model Info:")
    print(f"   - Parameters: {total_params:,}")
    print(f"   - File Size: {model_size_mb:.2f} MB")
    
    # Warm up
    print("🔥 Warming up model...")
    warmup_batch = x_test[:min(32, len(x_test))]
    _ = model.predict(warmup_batch, verbose=0)
    
    # Timing benchmark
    print("⏱️ Running timing benchmark...")
    times = []
    batch_size = 1  # Single image inference
    num_iterations = min(100, len(x_test))
    
    for i in range(num_iterations):
        img = x_test[i:i+1]
        
        start_time = time.time()
        pred = model.predict(img, verbose=0)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(inference_time)
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{num_iterations}")
    
    # Timing statistics
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    p95_ms = np.percentile(times, 95)
    min_ms = np.min(times)
    max_ms = np.max(times)
    throughput_ips = 1000 / mean_ms  # Images per second
    
    print(f"⚡ Timing Results:")
    print(f"   - Mean: {mean_ms:.2f} ms")
    print(f"   - Std Dev: {std_ms:.2f} ms") 
    print(f"   - P95: {p95_ms:.2f} ms")
    print(f"   - Min: {min_ms:.2f} ms")
    print(f"   - Max: {max_ms:.2f} ms")
    print(f"   - Throughput: {throughput_ips:.1f} images/sec")
    
    # Accuracy benchmark
    print("🎯 Running accuracy benchmark...")
    all_predictions = model.predict(x_test, batch_size=32, verbose=0)
    predicted_classes = np.argmax(all_predictions, axis=1)
    true_classes = y_test.flatten()
    
    # Overall accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    
    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            per_class_accuracy[class_name] = class_acc
    
    # Confidence metrics
    confidences = np.max(all_predictions, axis=1)
    avg_confidence = np.mean(confidences)
    confidence_std = np.std(confidences)
    
    # Top-3 accuracy
    top3_predictions = np.argsort(all_predictions, axis=1)[:, -3:]
    top3_accuracy = np.mean([true_classes[i] in top3_predictions[i] for i in range(len(true_classes))])
    
    print(f"🎯 Accuracy Results:")
    print(f"   - Overall Accuracy: {accuracy*100:.2f}%")
    print(f"   - Top-3 Accuracy: {top3_accuracy*100:.2f}%")
    print(f"   - Average Confidence: {avg_confidence:.3f}")
    print(f"   - Confidence Std: {confidence_std:.3f}")
    
    print(f"📊 Per-Class Accuracy:")
    for class_name, class_acc in per_class_accuracy.items():
        print(f"   - {class_name}: {class_acc*100:.1f}%")
    
    # Create summary dictionary
    summary = {
        "model_name": model_name,
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "throughput_ips": throughput_ips,
        "params": total_params,
        "model_size_mb": model_size_mb,
        "avg_confidence": avg_confidence,
        "confidence_std": confidence_std,
        "n_samples": len(x_test),
        "per_class_accuracy": per_class_accuracy
    }
    
    # Save individual results
    model_dir = os.path.join(RESULTS_ROOT, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(model_dir, f"{model_name.lower()}_summary.json")
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {summary_file}")
    print(f"✅ {model_name} benchmark complete!\n")
    
    return summary

def run_all_benchmarks():
    """Run benchmarks for all models"""
    print("🚀 STARTING COMPREHENSIVE MODEL BENCHMARKS")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load test data
    num_samples = 1000  # Adjust as needed
    x_test, y_test = load_and_preprocess_data(num_samples)
    
    # Run benchmarks for each model
    all_results = []
    
    for model_name, model_path in MODEL_FILES.items():
        if os.path.exists(model_path):
            result = benchmark_model(model_name, model_path, x_test, y_test)
            if result:
                all_results.append(result)
        else:
            print(f"❌ Model file not found: {model_path}")
    
    # Print comprehensive comparison
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE BENCHMARK COMPARISON")
    print("=" * 80)
    
    if all_results:
        # Create comparison DataFrame
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                "Model": result["model_name"],
                "Accuracy (%)": f"{result['accuracy']*100:.2f}",
                "Top-3 Acc (%)": f"{result['top3_accuracy']*100:.2f}",
                "Mean Time (ms)": f"{result['mean_ms']:.2f}",
                "P95 Time (ms)": f"{result['p95_ms']:.2f}",
                "Throughput (img/s)": f"{result['throughput_ips']:.1f}",
                "Model Size (MB)": f"{result['model_size_mb']:.2f}",
                "Parameters": f"{result['params']:,}",
                "Avg Confidence": f"{result['avg_confidence']:.3f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Speed comparison
        print(f"\n⚡ SPEED RANKING:")
        speed_sorted = sorted(all_results, key=lambda x: x['mean_ms'])
        for i, result in enumerate(speed_sorted, 1):
            print(f"{i}. {result['model_name']}: {result['mean_ms']:.2f} ms")
        
        # Accuracy comparison  
        print(f"\n🎯 ACCURACY RANKING:")
        acc_sorted = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        for i, result in enumerate(acc_sorted, 1):
            print(f"{i}. {result['model_name']}: {result['accuracy']*100:.2f}%")
        
        # Efficiency comparison (accuracy per ms)
        print(f"\n⚖️ EFFICIENCY RANKING (Accuracy/Time):")
        eff_sorted = sorted(all_results, key=lambda x: x['accuracy']/x['mean_ms'], reverse=True)
        for i, result in enumerate(eff_sorted, 1):
            efficiency = result['accuracy'] / result['mean_ms'] * 1000
            print(f"{i}. {result['model_name']}: {efficiency:.2f} acc/sec")
        
        # Size comparison
        print(f"\n💾 SIZE RANKING:")
        size_sorted = sorted(all_results, key=lambda x: x['model_size_mb'])
        for i, result in enumerate(size_sorted, 1):
            print(f"{i}. {result['model_name']}: {result['model_size_mb']:.2f} MB")
        
        # Save aggregated results
        os.makedirs(os.path.join(RESULTS_ROOT, "aggregated"), exist_ok=True)
        
        df.to_csv(os.path.join(RESULTS_ROOT, "aggregated", "comparison_table.csv"), index=False)
        
        with open(os.path.join(RESULTS_ROOT, "aggregated", "all_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 All results saved to: {RESULTS_ROOT}/aggregated/")
        print("🎉 BENCHMARK COMPLETE!")
        
        return all_results
    else:
        print("❌ No successful benchmarks completed!")
        return []

if __name__ == "__main__":
    print("Choose benchmark option:")
    print("1. Run fresh benchmarks (recommended)")
    print("2. Aggregate existing results only")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run fresh benchmarks
        results = run_all_benchmarks()
    else:
        # Original aggregation code
        rows = []
        for jf in glob.glob(os.path.join(RESULTS_ROOT, "*", "*_summary.json")):
            with open(jf) as f:
                s = json.load(f)
            name = s.get("model_name", os.path.basename(os.path.dirname(jf)))
            row = {
                "model": name,
                "accuracy": s.get("accuracy"),
                "mean_ms": s.get("mean_ms"),
                "p95_ms": s.get("p95_ms"),
                "throughput_ips": s.get("throughput_ips"),
                "params": s.get("params"),
                "n_samples": s.get("n_samples"),
                "summary_json": jf
            }
            model_path = MODEL_FILES.get(name)
            if model_path and os.path.exists(model_path):
                row["model_size_mb"] = os.path.getsize(model_path) / (1024*1024)
            else:
                row["model_size_mb"] = None
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows).sort_values("model")
            print(df.to_string(index=False))
        else:
            print("No existing results found. Run option 1 to generate benchmarks.")