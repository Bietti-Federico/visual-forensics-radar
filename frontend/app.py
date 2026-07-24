# app.py

import streamlit as st
import requests
import os

API_URL = os.getenv("DEEPFAKE_API_URL", "http://127.0.0.1:8000/analyze")
BATCH_API_URL = os.getenv("DEEPFAKE_API_BATCH_URL", API_URL.replace("/analyze", "/analyze-batch"))
MAX_BATCH_SIZE = 4

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
    st.markdown("-**OCR** (Text Integrity)")
    st.markdown("-**Fast Triage Scoring** (Score + Band)")
    st.markdown(f"-**Batch Size:** {MAX_BATCH_SIZE}")
    st.markdown("---")

st.markdown('<p class="main-header"> Deepfake & AI Forensics Radar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload one or more suspicious images to run a fast, batch triage analysis.</p>', unsafe_allow_html=True)


uploaded_files = st.file_uploader("Upload Target Image(s) (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def render_analysis_result(index, uploaded_file, data):
    decision = data.get("final_decision", {})
    diags = data.get("diagnostics", {})
    score = decision.get("score")
    risk_label = decision.get("risk_label", "N/A")
    risk_band = decision.get("risk_band", "").lower()
    component_scores = decision.get("component_scores", {})
    analysis_route = data.get("analysis_route", "unknown")
    document_type = data.get("document_type", "unknown")
    document_confidence = data.get("document_type_confidence", 0)
    extracted_fields = diags.get("extracted_fields", {})
    type_info = diags.get("document_type_classification", {})

    title_score = f"{score:.1f}" if isinstance(score, (int, float)) else "N/A"
    with st.expander(f"{index + 1}. {uploaded_file.name} • {document_type} • {risk_label} • {title_score}", expanded=index < 2):
        col_img, col_res = st.columns([2, 3], gap="large")

        with col_img:
            st.image(uploaded_file, width="stretch")
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "MIME": uploaded_file.type,
            }
            st.json(file_details)

        with col_res:
            st.subheader("Executive Diagnosis")
            st.metric("Document Type", document_type)
            st.metric("Type Confidence", f"{document_confidence:.1f}%")
            st.write(f"**Route:** {analysis_route}")

            if isinstance(score, (int, float)):
                st.metric("Suspicion Score", f"{score:.1f}/100")
                st.metric("Risk Band", risk_label)
                st.progress(int(min(max(score, 0), 100)))

                if "altamente" in risk_band:
                    st.error(decision.get("risk_label", "Altamente sospechoso"))
                elif "sospechoso" in risk_band:
                    st.warning(decision.get("risk_label", "Sospechoso"))
                else:
                    st.success(decision.get("risk_label", "Poco sospechoso"))
            else:
                st.info(risk_label)

            evidence = decision.get("evidence", [])
            if evidence:
                st.markdown("**Evidence:**")
                for item in evidence:
                    st.write(f"- {item}")

        st.markdown("---")
        if analysis_route == "receipt_control":
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["ELA (Pixels)", "CLIP (Semantics)", "Metadata", "OCR", "Score Breakdown"])

            with tab1:
                ela_data = diags.get("ela_layer", {})
                st.markdown("#### Error Level Analysis")
                st.caption("Detects compression differences and potential Photoshop edits.")
                col1, col2 = st.columns(2)
                col1.metric("Max Pixel Difference", f"{ela_data.get('max_difference', 0)}")
                col2.metric("ELA Threshold", f"{ela_data.get('threshold_used', 0)}")
                if ela_data.get("anomaly_detected"):
                    st.error("Status: Anomaly Detected")
                else:
                    st.success("Status: Clean")

            with tab2:
                clip_data = diags.get("clip_layer", {})
                st.markdown("#### Routing Signal")
                st.caption("CLIP is used here as a soft routing/support signal, not as a fraud verdict.")
                st.metric("Document Routing Confidence", f"{document_confidence:.1f}%")
                st.progress(int(min(max(document_confidence, 0), 100)))
                st.write(f"**Best label:** {type_info.get('document_type_label', 'N/A')}")
                st.write(f"**CLIP support score:** {component_scores.get('clip', 0):.1f}")

            with tab3:
                metadata_data = diags.get("metadata_layer", {})
                st.markdown("#### Metadata Inspection")
                st.caption("Looks for EXIF, software tags, and basic file anomalies.")
                st.metric("Metadata Score", f"{component_scores.get('metadata', 0):.1f}")
                st.write(f"**Format:** {metadata_data.get('format', 'N/A')}")
                st.write(f"**Size:** {metadata_data.get('file_size_bytes', 0)} bytes")
                st.write(f"**Resolution:** {metadata_data.get('width', 0)} x {metadata_data.get('height', 0)}")
                st.write(f"**Has EXIF:** {metadata_data.get('has_exif', False)}")
                software = metadata_data.get("software")
                if software:
                    st.write(f"**Software:** {software}")

            with tab4:
                ocr_data = diags.get("ocr_layer", {})
                local_ela_regions = diags.get("ocr_local_ela_regions", [])
                receipt_consistency = ocr_data.get("receipt_consistency", {})
                st.markdown("#### OCR & Local Integrity")
                st.caption("Reads document text and checks numeric regions with local ELA.")
                st.metric("OCR Suspicion Score", f"{component_scores.get('ocr', 0):.1f}")
                st.metric("OCR Richness", f"{ocr_data.get('richness_score', 0)}")
                st.metric("Receipt Consistency", f"{component_scores.get('receipt_consistency', 0):.1f}")
                st.write(f"**Mean confidence:** {ocr_data.get('mean_confidence', 0):.1f}%")
                st.write(f"**Word count:** {ocr_data.get('word_count', 0)}")
                st.write(f"**Numeric tokens:** {ocr_data.get('numeric_token_count', 0)}")
                if ocr_data.get("keyword_hits"):
                    st.write(f"**Keywords:** {', '.join(ocr_data.get('keyword_hits', []))}")

                if receipt_consistency:
                    st.markdown("**Receipt consistency:**")
                    st.write(f"**Detected:** {receipt_consistency.get('detected', False)}")
                    st.write(f"**Consistency score:** {receipt_consistency.get('consistency_score', 0)}")
                    fields = receipt_consistency.get("fields", {})
                    if fields:
                        st.json(fields)
                    signals = receipt_consistency.get("signals", [])
                    if signals:
                        for signal in signals:
                            st.write(f"- {signal}")

                if local_ela_regions:
                    st.markdown("**Local ELA on OCR candidates:**")
                    for region in local_ela_regions:
                        text = region.get("text", "") or "[sin texto]"
                        diff = region.get("max_difference", 0)
                        anomaly = region.get("anomaly_detected", False)
                        st.write(f"- {text}: diff={diff} | anomaly={anomaly}")
                else:
                    st.info("No OCR candidate regions were detected for local ELA.")

            with tab5:
                st.markdown("#### Score Breakdown")
                st.caption("Weighted components used to compute the final triage score.")
                st.metric("Final Score", f"{score:.1f}")
                st.write(f"**Local ELA:** {component_scores.get('local_ela', 0):.1f}")
                st.write(f"**OCR:** {component_scores.get('ocr', 0):.1f}")
                st.write(f"**Global ELA:** {component_scores.get('global_ela', 0):.1f}")
                st.write(f"**Metadata:** {component_scores.get('metadata', 0):.1f}")
                st.write(f"**CLIP:** {component_scores.get('clip', 0):.1f}")
        else:
            tab1, tab2 = st.tabs(["Categorization", "Raw Signals"])

            with tab1:
                st.markdown("#### Categorization")
                st.write(f"**Route:** {analysis_route}")
                st.write(f"**Document type:** {document_type}")
                st.write(f"**Confidence:** {document_confidence:.1f}%")

                if document_type == "dni_front":
                    st.write("Documento identificado como DNI frente.")
                elif document_type == "dni_back":
                    st.write("Documento identificado como DNI dorso.")
                elif document_type == "card":
                    st.write("Documento identificado como tarjeta.")
                elif document_type == "homebanking":
                    st.write("Documento identificado como homebanking.")
                else:
                    st.write("Documento categorizado sin control de fraude.")

            with tab2:
                st.markdown("#### Raw Signals")
                st.write(f"**Type label:** {type_info.get('document_type_label', 'N/A')}")
                st.json(type_info.get("document_type_probabilities", {}))

        if analysis_route != "receipt_control":
            st.markdown("---")
            st.subheader("Extraction Details")
            extraction_col1, extraction_col2 = st.columns(2)

            with extraction_col1:
                st.markdown("#### Document Extraction")
                st.caption("No fraud control is applied to this type of document.")
                if document_type == "dni_front":
                    st.write(f"**DNI Trámite:** {extracted_fields.get('dni_tramite', 'No detectado')}")
                    st.write(f"**DNI Number:** {extracted_fields.get('dni_number', 'No detectado')}")
                    st.write("**Side:** Frente")
                elif document_type == "dni_back":
                    st.write(f"**DNI Number:** {extracted_fields.get('dni_number', 'No detectado')}")
                    st.write("**Side:** Dorso")
                elif document_type == "card":
                    st.write(f"**Bank:** {extracted_fields.get('bank_name', 'No detectado')}")
                    card_numbers = extracted_fields.get('card_numbers', [])
                    masked_numbers = extracted_fields.get('masked_card_numbers', [])
                    st.write(f"**Card Numbers:** {', '.join(card_numbers) if card_numbers else 'No detectado'}")
                    if masked_numbers:
                        st.write(f"**Masked Numbers:** {', '.join(masked_numbers)}")
                else:
                    st.write("Documento categorizado sin control de fraude.")

            with extraction_col2:
                ocr_data = diags.get("ocr_layer", {})
                st.markdown("#### OCR Summary")
                st.write(f"**Mean confidence:** {ocr_data.get('mean_confidence', 0):.1f}%")
                st.write(f"**Word count:** {ocr_data.get('word_count', 0)}")
                st.write(f"**Richness score:** {ocr_data.get('richness_score', 0)}")
                st.write(f"**Text excerpt:** {ocr_data.get('text_excerpt', '')}")


if uploaded_files:
    if len(uploaded_files) > MAX_BATCH_SIZE:
        st.info(f"Selected {len(uploaded_files)} images. They will be processed in batches of {MAX_BATCH_SIZE}.")

    analyze_btn = st.button("Run Forensic Analysis", width="stretch", type="primary")

    if analyze_btn:
        with st.spinner("Analyzing batch... Please wait."):
            try:
                all_results = []
                summary_rows = []

                for start_idx in range(0, len(uploaded_files), MAX_BATCH_SIZE):
                    batch = uploaded_files[start_idx:start_idx + MAX_BATCH_SIZE]
                    multipart_files = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in batch
                    ]

                    response = requests.post(BATCH_API_URL, files=multipart_files)
                    if response.status_code != 200:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        st.stop()

                    payload = response.json()
                    batch_results = payload.get("results", [])

                    for file_obj, result in zip(batch, batch_results):
                        all_results.append((file_obj, result))
                        decision = result.get("final_decision", {})
                        summary_rows.append({
                            "file": file_obj.name,
                            "type": result.get("document_type", "unknown"),
                            "route": result.get("analysis_route", "unknown"),
                            "score": decision.get("score", "N/A"),
                            "risk": decision.get("risk_label", "N/A"),
                            "band": decision.get("risk_band", "N/A"),
                        })

                if summary_rows:
                    st.subheader("Batch Summary")
                    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

                st.subheader("Detailed Analyses")
                for idx, (file_obj, result) in enumerate(all_results):
                    render_analysis_result(idx, file_obj, result)

            except requests.exceptions.ConnectionError:
                st.error("ERROR: Could not connect to the Backend API.")