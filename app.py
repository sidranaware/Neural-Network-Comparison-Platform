# app.py

import os
os.environ["TF_METAL_DISABLE"] = "1"
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import time
import random
import json

# Import utilities
from utils.model_loader import load_models, MODEL_FILES, CLASS_NAMES
from utils.data_processor import preprocess_image, analyze_prediction_confidence

# Import tab modules
from tabs.tab1_overview import render_overview_tab
from tabs.tab2_comparison import render_comparison_tab
from tabs.tab3_analysis import render_analysis_tab
from tabs.tab4_insights import render_insights_tab
from tabs.tab5_benchmarks import render_benchmarks_tab

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'
tf.config.experimental.enable_op_determinism()

# Initialize session state
if 'all_results' not in st.session_state:
    st.session_state.all_results = []

# Page configuration
st.set_page_config(page_title="CNN vs BNN vs Hybrid Demo", layout="wide")

# Header with subtle gradient
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
           padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🧠 Neural Network Comparison Platform</h1>
    <h3 style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-weight: 300;">CNN vs BNN vs Hybrid Performance Analysis</h3>
</div>
""", unsafe_allow_html=True)

# Custom CSS for subtle UI improvements
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: none; border-radius: 15px; padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    [data-testid="metric-container"] > div, [data-testid="metric-container"] label {
        color: white !important; font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Options")

# Clear button
if st.sidebar.button("🗑️ Clear All Results"):
    st.session_state.all_results = []
    st.rerun()

# Show statistics in sidebar
if st.session_state.all_results:
    total_images = len(set([r['file'] for r in st.session_state.all_results]))
    total_predictions = len(st.session_state.all_results)
    st.sidebar.metric("📊 Total Images", total_images)
    st.sidebar.metric("🔍 Total Predictions", total_predictions)

# File upload and model selection
uploaded_files = st.sidebar.file_uploader("📁 Upload Images (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
selected_models = st.sidebar.multiselect("🤖 Select Model Variant(s)", options=list(MODEL_FILES.keys()))

# Validate uploads
valid_images = []
if uploaded_files:
    for f in uploaded_files:
        if f.name.lower().endswith((".png", ".jpg", ".jpeg")):
            valid_images.append(f)

if valid_images:
    st.sidebar.success(f"✅ {len(valid_images)} valid image(s) ready")

# Main content
if valid_images and selected_models:
    models_dict = load_models()
    processed_files = set([r['file'] for r in st.session_state.all_results])
    new_images = [f for f in valid_images if f.name not in processed_files]
    
    if new_images:
        st.markdown(f"### 🔄 Processing {len(new_images)} new image(s)...")
        cols = st.columns(min(4, len(new_images)))
        for idx, f in enumerate(new_images):
            with cols[idx % 4]:
                img = Image.open(f).convert("RGB")
                st.image(img, width=120, caption=f.name)
                x = preprocess_image(img)
                
                result = {"file": f.name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                
                # Prediction and analysis
                for name in selected_models:
                    mdl = models_dict.get(name)
                    if mdl:
                        start = time.time()
                        pred = mdl.predict(x, verbose=0)[0]
                        elapsed = (time.time() - start) * 1000
                        
                        # Simple prediction without detailed analysis
                        label = CLASS_NAMES[int(np.argmax(pred))]
                        conf = float(np.max(pred))
                        
                        result[f"{name}_predicted"] = label
                        result[f"{name}_confidence"] = f"{conf:.2f}"
                        result[f"{name}_time_ms"] = elapsed
                
                st.session_state.all_results.append(result)
        st.success(f"✅ Successfully processed {len(new_images)} new image(s)!")
    else:
        st.info("ℹ️ All uploaded images have already been processed.")
    
    if st.session_state.all_results:
        df = pd.DataFrame(st.session_state.all_results)
        st.markdown("### 📝 All Inference Results")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download All Results CSV", data=csv, 
                          file_name=f"all_results_{time.strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        
        # Analytics Dashboard
        st.markdown("---")
        st.markdown("## 📊 Analytics Dashboard")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Comparison", "🔬 Analysis", "💡 Insights", "📈 Benchmarks"])
        
        with tab1:
            render_overview_tab(df, selected_models)
        
        with tab2:
            render_comparison_tab(df, selected_models)
        
        with tab3:
            render_analysis_tab(df, selected_models)
        
        with tab4:
            render_insights_tab(df, selected_models)
        
        with tab5:
            render_benchmarks_tab(df, selected_models)

elif st.session_state.all_results:
    # Show existing results even when no new upload
    df = pd.DataFrame(st.session_state.all_results)
    st.markdown("### 📝 Previous Results")
    st.dataframe(df, use_container_width=True)
    
    # Download option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Previous Results",
        data=csv,
        file_name=f"previous_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Performance chart for previous data
    st.markdown("### ⚡ Previous Performance")
    perf_data = []
    time_cols = [col for col in df.columns if col.endswith('_time_ms')]
    
    for col in time_cols:
        model_name = col.replace('_time_ms', '')
        avg_time = df[col].mean()
        perf_data.append({"model": model_name, "avg_time_ms": avg_time})
    
    if perf_data:
        perf_df = pd.DataFrame(perf_data)
        st.bar_chart(data=perf_df, x="model", y="avg_time_ms")
    
    st.info("Upload new images to add to the results!")
else:
    st.info("Upload an image and select a model to get started.")
