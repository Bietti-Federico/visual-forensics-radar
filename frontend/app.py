# app.py

import streamlit as st
import requests
import os

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
    st.markdown("-**OCR** (Text Integrity)")
    st.markdown("-**Typography** (Font Consistency)")
    st.markdown("-**Fast Triage Scoring** (Score + Band)")
    st.markdown("---")

st.markdown('<p class="main-header"> Deepfake & AI Forensics Radar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a suspicious image to run a fast triage analysis.</p>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload Target Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

def build_audit_summary(file_name, data):
    """
    Condenses everything scattered across the tabs (evidence, flagged fields, receipt
    consistency) into one compact, copy-pasteable block — so a reviewer doesn't have to
    click through 5 tabs to get the gist of why a document was flagged.
    """
    decision = data.get("final_decision", {})
    diags = data.get("diagnostics", {})
    analysis_route = data.get("analysis_route", "unknown")
    document_type = data.get("document_type", "unknown")
    document_confidence = data.get("document_type_confidence", 0)

    lines = [f"INFORME DE AUDITORÍA — {file_name}", "=" * 50]
    lines.append(f"Tipo de documento: {document_type} ({document_confidence:.1f}%)")
    lines.append(f"Ruta de análisis: {analysis_route}")

    if analysis_route != "receipt_control":
        lines.append("Este tipo de documento no recibe control de fraude (solo categorización).")
        return "\n".join(lines)

    score = decision.get("score")
    score_text = f"{score:.1f}/100" if isinstance(score, (int, float)) else "N/A"
    lines.append(f"Score de sospecha: {score_text} — {decision.get('risk_label', 'N/A')}")

    timings = data.get("timings", {})
    if timings.get("total_s") is not None:
        lines.append(f"Tiempo total de análisis: {timings['total_s']:.2f}s")
    lines.append("")

    primary_reason = decision.get("primary_reason")
    if primary_reason:
        lines.append("MOTIVO PRINCIPAL:")
        lines.append(f"  {primary_reason}")
        lines.append("")

    evidence = decision.get("evidence", [])
    if evidence:
        lines.append("EVIDENCIA (con aporte al scoring):")
        for item in evidence:
            text = item.get("text", "") if isinstance(item, dict) else item
            aporte = item.get("aporte") if isinstance(item, dict) else None
            lines.append(f"  - {text}")
            if aporte:
                lines.append(f"      ↳ {aporte}")
        lines.append("")

    ela_hits = [r for r in diags.get("ocr_local_ela_regions", []) if r.get("anomaly_detected")]
    if ela_hits:
        lines.append("CAMPOS SEÑALADOS POR ELA LOCAL:")
        lines.extend(
            f"  - '{region.get('text', '')}' (diff={region.get('max_difference', 0)}, "
            f"z relativo={region.get('relative_z_score')})"
            for region in ela_hits
        )
        lines.append("")

    anomalous_fields = diags.get("typography_layer", {}).get("anomalous_fields", [])
    if anomalous_fields:
        lines.append("CAMPOS SEÑALADOS POR TIPOGRAFÍA:")
        lines.extend(
            f"  - '{field.get('text', '')}' — señal dominante: {field.get('dominant_feature')} "
            f"(z={field.get('max_abs_z')})"
            for field in anomalous_fields
        )
        lines.append("")

    consistency_signals = diags.get("ocr_layer", {}).get("receipt_consistency", {}).get("signals", [])
    if consistency_signals:
        lines.append("CONSISTENCIA DEL RECIBO:")
        lines.extend(f"  - {signal}" for signal in consistency_signals)
        lines.append("")

    return "\n".join(lines).rstrip()


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
    timings = data.get("timings", {})

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
            primary_reason = decision.get("primary_reason")
            if primary_reason:
                if "confirmada" in primary_reason.lower() or "altamente" in risk_band:
                    st.error(primary_reason)
                elif "sospechoso" in risk_band:
                    st.warning(primary_reason)
                else:
                    st.info(primary_reason)

            st.subheader("Executive Diagnosis")
            st.metric("Document Type", document_type)
            st.metric("Type Confidence", f"{document_confidence:.1f}%")
            st.write(f"**Route:** {analysis_route}")
            if timings.get("total_s") is not None:
                st.metric("Analysis Time", f"{timings['total_s']:.2f}s")

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
                    text = item.get("text", "") if isinstance(item, dict) else item
                    aporte = item.get("aporte") if isinstance(item, dict) else None
                    if aporte:
                        st.write(f"- {text}")
                        st.caption(f"  ↳ {aporte}")
                    else:
                        st.write(f"- {text}")

        st.markdown("---")
        st.markdown("#### 📋 Resumen para auditoría")
        st.caption("Todo lo relevante en un solo bloque — el ícono de copiar está arriba a la derecha.")
        st.code(build_audit_summary(uploaded_file.name, data), language=None)

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
                st.markdown("#### Routing Signal")
                st.caption("CLIP is used only to route the document to the right pipeline, not as a fraud signal.")
                st.metric("Document Routing Confidence", f"{document_confidence:.1f}%")
                st.progress(int(min(max(document_confidence, 0), 100)))
                st.write(f"**Best label:** {type_info.get('document_type_label', 'N/A')}")

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

                capture_mode = metadata_data.get("capture_mode")
                if capture_mode:
                    label = "📷 Foto física (fondo/mesa visible)" if capture_mode == "photo" else "💻 Digital / escaneo (fondo plano)"
                    st.write(f"**Modo de captura detectado:** {label}")
                    st.caption(
                        f"distancia de esquina al blanco={metadata_data.get('max_corner_distance_from_white')} · "
                        f"saturación media={metadata_data.get('mean_saturation')} — "
                        "si se detecta foto física, tipografía relaja su umbral de anomalía (paper curvature, "
                        "sombras y ángulo agregan variación normal que no es edición)."
                    )

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

                with st.expander("Texto crudo extraído por OCR (para depurar campos faltantes)"):
                    st.caption(
                        "Si un campo no aparece en ELA local ni en tipografía, primero revisá acá si "
                        "el OCR lo leyó bien — si no está o está garbled, el problema es de lectura, "
                        "no de calibración de umbrales."
                    )
                    st.code(ocr_data.get("text_blob", "") or "(sin texto)", language=None)

                if ocr_data.get("keyword_hits"):
                    st.write(f"**Keywords:** {', '.join(ocr_data.get('keyword_hits', []))}")

                extracted_fields_ocr = ocr_data.get("extracted_fields", {})
                if any(extracted_fields_ocr.get(k) for k in ("cuil", "persona", "tipo_recibo", "periodo")):
                    st.markdown("**Campos extraídos (capa de extracción):**")
                    st.caption(
                        "Mismos extractores que la rama de extracción — visibles acá para depurar "
                        "ambos ciclos (fraude + extracción) sobre el mismo caso."
                    )
                    st.write(f"**Tipo de recibo:** {extracted_fields_ocr.get('tipo_recibo') or 'No detectado'}")
                    st.write(f"**Persona:** {extracted_fields_ocr.get('persona') or 'No detectado'}")
                    st.write(f"**Período:** {extracted_fields_ocr.get('periodo') or 'No detectado'}")
                    st.write(f"**CUIL:** {extracted_fields_ocr.get('cuil') or 'No detectado'}")

                line_items = ocr_data.get("line_items", [])
                if line_items:
                    st.markdown(f"**Montos por ítem de línea ({len(line_items)}):**")
                    st.table([
                        {"Concepto": item.get("concepto", ""), "Monto": item.get("monto"), "Raw": item.get("raw", "")}
                        for item in line_items
                    ])

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
                    st.caption(
                        "relative_z compares each field's diff against the rest of this same "
                        "document (null = too few comparable fields to trust that baseline)."
                    )
                    for region in local_ela_regions:
                        text = region.get("text", "") or "[sin texto]"
                        diff = region.get("max_difference", 0)
                        anomaly = region.get("anomaly_detected", False)
                        relative_z = region.get("relative_z_score")
                        highlight_marker = (
                            f" 🎨fondo resaltado (medido solo sobre tinta, sin comparación relativa)"
                            if region.get("background_highlighted") else ""
                        )
                        st.write(f"- {text}: diff={diff} | anomaly={anomaly} | relative_z={relative_z}{highlight_marker}")
                else:
                    st.info("No OCR candidate regions were detected for local ELA.")

                typography_data = diags.get("typography_layer", {})
                buckets = typography_data.get("buckets", {})
                anomalous_fields = typography_data.get("anomalous_fields", [])
                st.markdown("**Typography consistency:**")
                st.caption("Compares glyph height, ink density, slant and letter proportions of each field against the rest of the document.")
                if typography_data.get("capture_mode"):
                    mode_label = "foto física (umbral relajado)" if typography_data.get("capture_mode") == "photo" else "digital/escaneo (umbral estándar)"
                    st.caption(f"Modo de captura: {mode_label} · z-score umbral usado: {typography_data.get('z_threshold_used')}")

                with st.expander("¿Qué mide esta validación?"):
                    st.markdown(
                        """
Para cada monto y fecha detectado por OCR, se recorta el campo y se miden 4 señales
independientes:
- **Altura del glifo** (`height`): el alto en píxeles del texto.
- **Densidad de tinta** (`ink_ratio`): qué fracción del recorte es trazo de texto vs. fondo,
  detectando automáticamente si el texto es oscuro sobre claro o claro sobre oscuro
  (por ejemplo una fila de "Total" resaltada en homebanking).
- **Inclinación / caligrafía** (`slant_angle`): el ángulo dominante del trazo, medido
  **por carácter individual** (cada dígito/letra se segmenta como su propia forma
  conectada de tinta) y luego se toma la mediana entre los caracteres del campo —
  distingue una fuente itálica, cursiva o manuscrita insertada en medio de texto
  impreso derecho, sin diluirse por cuántos caracteres tiene el valor.
- **Proporción de letra** (`aspect_ratio`): ancho/alto del recuadro ajustado a la tinta
  de **cada carácter** (no del campo completo) — así un monto de muchos dígitos no
  aparenta ser "más ancho" solo por tener más dígitos; lo que se compara es la forma
  real de la letra.

Estas 4 métricas se comparan **contra los demás campos del mismo tipo en el mismo
documento** (montos contra montos, fechas contra fechas — nunca un monto contra una
fecha, ni un número de cuenta contra un monto), usando mediana y desviación absoluta
mediana (MAD) de forma robusta: cada campo se compara contra el resto excluyéndose a
sí mismo, para que un solo valor atípico no contamine su propia referencia.

Un campo se marca como **anómalo** si alguna de las 4 señales supera un z-score de
3.5 — un umbral estadístico exigente (equivalente a una desviación muy poco probable
por azar). La señal con mayor z-score queda registrada como "señal dominante", para
saber si lo que se destacó fue el tamaño, la densidad, la inclinación o la
proporción de la letra. Esto es típico de un valor pegado o reescrito con otra
fuente/herramienta (o de un dato completado a mano) que sobrevive a la reimpresión o
al reescaneo — y que por eso el ELA de compresión JPEG no detecta.

**Límites a tener en cuenta:**
- Necesita una población mínima por bucket (5 montos o 3 fechas) para confiar en la
  comparación; si el documento tiene menos campos de un tipo, ese bucket queda
  `insufficient_population` y no genera señal.
- Un recorte sin contraste confiable (celda casi en blanco, muy baja separación entre
  texto y fondo) se descarta directamente en vez de sumar ruido.
- Inclinación y proporción necesitan al menos un carácter con suficientes píxeles de
  tinta para estimarse; si ningún carácter del campo alcanza ese mínimo, quedan en un
  valor neutro (0° / 1:1) y se excluyen de la comparación (no se tratan como una
  medición real ni contaminan la referencia de otros campos).
- Es una señal de **triage para orientar la revisión humana**, no una prueba de
  edición por sí sola — cuando coincide con una anomalía de ELA local en el mismo
  campo (misma posición), el sistema la marca como corroborada y sube el score con
  más peso; y si además hay una inconsistencia lógica (período o aritmética) en el
  mismo recibo, eso también suma un piso adicional de score.
                        """
                    )

                for bucket_name, bucket_info in buckets.items():
                    st.write(f"- Bucket `{bucket_name}`: {bucket_info.get('status')} (n={bucket_info.get('sample_count', 0)})")
                if anomalous_fields:
                    for field in anomalous_fields:
                        marker = "🔑" if field.get("is_key_field") else "•"
                        st.write(
                            f"{marker} {field.get('text', '')}: señal dominante=`{field.get('dominant_feature')}` "
                            f"(z-score={field.get('max_abs_z')}) | bucket={field.get('bucket')}"
                        )
                        z_scores = field.get("z_scores")
                        if z_scores:
                            st.caption(
                                " · ".join(f"{name}: {value}" for name, value in z_scores.items())
                            )
                else:
                    st.info("No se detectaron campos con tipografía atípica.")

            with tab5:
                st.markdown("#### Score Breakdown")
                st.caption("Weighted components used to compute the final triage score.")
                st.metric("Final Score", f"{score:.1f}")
                st.write(f"**Local ELA:** {component_scores.get('local_ela', 0):.1f}")
                st.write(f"**Typography:** {component_scores.get('typography', 0):.1f}")
                st.write(f"**OCR:** {component_scores.get('ocr', 0):.1f}")
                st.write(f"**Receipt Consistency:** {component_scores.get('receipt_consistency', 0):.1f}")
                st.write(f"**Global ELA:** {component_scores.get('global_ela', 0):.1f}")
                st.write(f"**Metadata:** {component_scores.get('metadata', 0):.1f} (informativo, no pesa en el score)")

                if timings:
                    st.markdown("---")
                    st.markdown("#### ⏱ Tiempos de análisis")
                    st.caption("Cuánto tardó cada etapa del pipeline, en segundos.")
                    st.metric("Tiempo total", f"{timings.get('total_s', 0):.2f}s")
                    stage_labels = {
                        "document_type_classification_s": "Clasificación de documento (CLIP)",
                        "ela_global_s": "ELA global",
                        "metadata_s": "Metadata",
                        "ocr_s": "OCR",
                        "local_ela_regions_s": "ELA local por campo",
                        "typography_s": "Tipografía",
                    }
                    for key, label in stage_labels.items():
                        if key in timings:
                            st.write(f"**{label}:** {timings[key]:.2f}s")
        else:
            tab1, tab2 = st.tabs(["Categorization", "Raw Signals"])

            with tab1:
                st.markdown("#### Categorization")
                st.write(f"**Route:** {analysis_route}")
                st.write(f"**Document type:** {document_type}")
                st.write(f"**Confidence:** {document_confidence:.1f}%")
                if timings.get("total_s") is not None:
                    st.write(f"**Analysis time:** {timings['total_s']:.2f}s")

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


if uploaded_file is not None:
    analyze_btn = st.button("Run Forensic Analysis", width="stretch", type="primary")

    if analyze_btn:
        with st.spinner("Analyzing... Please wait."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code != 200:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    st.stop()

                result = response.json()
                st.subheader("Analysis Result")
                render_analysis_result(0, uploaded_file, result)

            except requests.exceptions.ConnectionError:
                st.error("ERROR: Could not connect to the Backend API.")