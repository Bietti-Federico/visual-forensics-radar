# app.py

import streamlit as st
import requests
import os
from PIL import Image

API_URL = os.getenv("DEEPFAKE_API_URL", "http://127.0.0.1:8000/analyze")

st.set_page_config(
    page_title="Deepfake Forensics Radar", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #888888; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Core System")
    
    st.markdown("---")
    st.markdown("**Active Engines:**")
    st.markdown("-**ELA** (Pixel Anomaly)")
    st.markdown("-**CLIP** (Semantic Vision)")
    st.markdown("-**Qwen2** (Logic / VLM)")
    st.markdown("---")

st.markdown('<p class="main-header"> Deepfake & AI Forensics Radar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a suspicious image to run a multi-layered architectural analysis.</p>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload Target Image(JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    col_img, col_res = st.columns([2, 3], gap="large")
    
    with col_img:
        st.subheader("Target Subject")
        st.image(uploaded_file, width="stretch")
        
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.1f} KB"}
        st.json(file_details)
        
        analyze_btn = st.button("Run Forensic Analysis", width="stretch", type="primary")

    if analyze_btn:
        with col_res:
            with st.spinner("Analyzing pixels, semantics, and logic... Please wait."):
                try:

                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        decision = data.get("final_decision", {})
                        risk = decision.get("risk_level", "").lower()
                        diags = data.get("diagnostics", {})
                        
                        st.subheader("Executive Diagnosis")
                        
                        if "high" in risk or "critical" in risk:
                            st.error(f"**{decision.get('title')}**\n\n{decision.get('description')}")
                        elif "medium" in risk:
                            st.warning(f"**{decision.get('title')}**\n\n{decision.get('description')}")
                        else:
                            st.success(f"**{decision.get('title')}**\n\n{decision.get('description')}")

                        st.markdown("---")
                        st.subheader("Detailed Engine Diagnostics")
                        
                        tab1, tab2, tab3 = st.tabs(["ELA (Pixels)", "CLIP (Semantics)", "VLM (Logic)"])
                        
                        with tab1:
                            ela_data = diags.get("ela_layer", {})
                            st.markdown("#### Error Level Analysis")
                            st.caption("Detects compression differences and potential Photoshop edits.")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Max Pixel Difference", f"{ela_data.get('max_difference', 0)}")
                            
                            if ela_data.get("anomaly_detected"):
                                col2.error("Status: Anomaly Detected")
                                st.info("**Insight:** The pixel compression is not homogeneous. This strongly suggests the image was digitally altered after it was created.")
                            else:
                                col2.success("Status: Clean")
                                st.info("**Insight:** Pixels are homogeneous. No traces of manual splicing or post-editing found.")

                        with tab2:
                            clip_data = diags.get("clip_layer", {})
                            ai_prob = clip_data.get("detailed_probabilities", {}).get("ai_probability", 0)
                            st.markdown("#### Latent Space Detection")
                            st.caption("Reads the invisible semantic patterns left by AI models.")
                            
                            st.metric("Probability of AI Generation", f"{ai_prob:.1f}%")
                            st.progress(int(ai_prob))
                            if ai_prob > 50:
                                st.warning("High probability of synthetic generation patterns.")
                            else:
                                st.success("Patterns are consistent with a real photograph.")

                        with tab3:
                            vlm_report = diags.get("vlm_detective_report", "No report available.")
                            st.markdown("#### Visual Logic Reasoning")
                            st.caption("Qwen2 acts as a detective, looking for physical impossibilities.")
                            
                            st.markdown(f"> *{vlm_report}*")

                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("ERROR: Could not connect to the Backend API.")