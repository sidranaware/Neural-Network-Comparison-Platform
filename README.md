# Neural Network Comparison Platform (CNN vs BNN vs Hybrid)

## 1. Overview
Interactive Streamlit dashboard to compare three CIFAR-10 image classification models:
- CNN (baseline accuracy-focused)
- BNN (binarized for efficiency)
- Hybrid (balance between accuracy & efficiency)

Supports real-time inference on uploaded images + benchmark analysis (speed, confidence, consistency, size, energy proxy).

## 2. Key Features
- Upload images → multi-model inference
- Per-image prediction times & confidence
- Aggregated analytics (tabs)
- Benchmark tab using pre-computed results
- Export all inference results to CSV
- Model selection & session persistence

## 3. Directory Structure
```
.
├── app.py
├── cnn_model.py
├── bnn_model.py
├── hybrid_model.py
├── aggregate_results.py          (optional: builds benchmark dataset)
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── model_loader.py
│   └── data_processor.py
├── tabs/
│   ├── __init__.py
│   ├── tab1_overview.py
│   ├── tab2_comparison.py
│   ├── tab3_analysis.py
│   ├── tab4_insights.py
│   └── tab5_benchmarks.py
└── models/
    ├── cnn_baseline_final.h5
    ├── bnn_baseline_final1.h5
    └── hybrid_final.h5
```

## 4. Technology Stack
- Python, TensorFlow / Keras
- Larq (binarized layers)
- Streamlit (UI)
- NumPy, Pandas, Pillow
- Optional: Plotly / Matplotlib

## 5. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 6. Running the Dashboard
```bash
streamlit run app.py
```
Default opens at http://localhost:8501

## 7. Models
If `.h5` files missing, retrain:
```bash
python cnn_model.py
python bnn_model.py
python hybrid_model.py
```
Each script:
- Loads CIFAR-10
- Normalizes data
- Builds architecture
- Trains with callbacks
- Saves final model to /models

## 8. Model Design Summary
| Model  | Emphasis          | Traits                          |
|--------|-------------------|---------------------------------|
| CNN    | Accuracy          | Deeper conv + dense head        |
| BNN    | Efficiency        | Binary conv layers after stem   |
| Hybrid | Trade-off         | FP stem + binary core + FP head |

## 9. Inference Flow (Runtime)
1. User uploads image(s)
2. Preprocessing: resize → normalize → batch
3. For each selected model: load (cached) → predict → record:
   - Class
   - Confidence
   - Inference time (ms)
4. Append record to session dataframe
5. Tabs render aggregated metrics

## 10. Tabs Overview
- Tab 1 Overview: totals, fastest model, consistency indicators
- Tab 2 Comparison: per-model aggregated stats
- Tab 3 Analysis: architecture & trade-off visuals
- Tab 4 Insights: energy proxy & recommendations
- Tab 5 Benchmarks: pre-computed results (from aggregate_results.py)

## 11. Benchmark Data
`aggregate_results.py` (if used) batches inference over a test subset and writes aggregated metrics (accuracy, avg time, std, model size). Output consumed by tab5.

## 12. Energy / Efficiency Proxy
Approximate efficiency via:
- Model size (parameters)
- Inference latency
- (Optional future) MAC count / binary op ratio

(No hardware wattmeter—stated as limitation.)

## 13. Reproducibility
Seeds set (Python, NumPy, TF). Deterministic ops enabled where supported:
```python
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

## 14. Extending
- Add more datasets (CIFAR-100, TinyImageNet)
- Add pruning / quantization aware training
- Replace energy proxy with real measurements
- Add batch benchmarking UI
- Confusion matrix & per-class charts

## 15. Limitations
- CIFAR-10 only
- Single-image inference path (no batching UI)
- Energy ≈ proxy, not direct measurement
- BNN accuracy gap vs full precision
- Not optimized for mobile deployment yet

## 16. Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| OOM during training | Large batch | Reduce batch size |
| Metal plugin warnings (macOS) | TF + GPU backend | `TF_METAL_DISABLE=1` already set |
| Slow first inference | Model load cold start | Keep models cached in session |
 

## 17. License
(Add: MIT / Apache-2.0 / Proprietary as applicable.)

## 18. Quick Usage Example
Upload an image → select models → view predictions → download CSV → explore tabs.

## 19. Contact / Maintainer
Author: Siddhiraj Ranaware
Institution: University of Liverpool
Email: sgsranaw@liverpool.co.uk

---