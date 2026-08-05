# app.py

import streamlit as st
import requests
import os

API_URL = os.getenv("PDF_METADATA_API_URL", "http://127.0.0.1:8000/analyze")

st.set_page_config(
    page_title="PDF Metadata Inspector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("### Core System")
    st.markdown("---")
    st.caption(
        "Analiza únicamente PDFs descargados directamente de la página oficial del "
        "banco/organismo. No hace OCR ni emite veredicto de fraude — muestra todo lo "
        "que el archivo PDF en sí mismo revela, para decidir qué es controlable por "
        "banco/tipo de documento."
    )

st.markdown("# 🔎 PDF Metadata Inspector")
st.markdown("Subí un PDF para ver toda la metadata que se puede extraer del archivo.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])


def render_markdown_table(rows: list[dict]) -> None:
    """
    Renders a list of dicts as a markdown pipe table via st.markdown instead of
    st.table/st.dataframe. Both of those go through a pandas DataFrame -> pyarrow
    serialization step to talk to the frontend, and a column with mixed types
    (bool/int/None, exactly what these rows contain) has been linked to native
    pyarrow crashes in slim Docker images — consistent with the delayed
    (GC-triggered, not synchronous) segfault seen after a couple of analyses.
    Building the table as plain markdown text avoids that native path entirely.
    """
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    st.markdown("\n".join(lines))


def render_result(uploaded_file, data):
    if data.get("status") != "success":
        st.error(f"Error al analizar el PDF: {data.get('message', 'desconocido')}")
        return

    file_info = data.get("file", {})
    meta = data.get("standard_metadata", {})
    structure = data.get("structure", {})
    text_layer = data.get("text_layer", {})
    signatures = data.get("signatures", {})
    embedded_files = data.get("embedded_files", [])
    xmp = data.get("xmp_metadata", {})

    tabs = st.tabs([
        "Resumen", "Metadata estándar", "Estructura / revisiones",
        "Capa de texto", "Firmas", "Archivos embebidos", "XMP crudo",
    ])

    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Páginas", structure.get("page_count", "?"))
        col2.metric("Tamaño", f"{file_info.get('size_bytes', 0) / 1024:.1f} KB")
        col3.metric("¿Tiene capa de texto?", "Sí" if text_layer.get("has_text_layer") else "No")
        col4.metric("¿Firmado?", "Sí" if signatures.get("has_signature_fields") else "No")

        if structure.get("likely_incrementally_updated"):
            st.warning(
                f"El PDF tiene {structure.get('eof_marker_count')} marcadores '%%EOF' — más de "
                "una actualización incremental además del guardado inicial. Los recibos de "
                "referencia (ANSES, municipalidades) tienen normalmente 2; esto es un tercero "
                "o más."
            )
        else:
            st.success(
                f"{structure.get('eof_marker_count')} marcador(es) '%%EOF' — dentro de lo esperado "
                "(el guardado inicial y, si corresponde, uno más por firma/finalización)."
            )

        gap_seconds = meta.get("creation_to_modification_gap_seconds")
        if gap_seconds is not None:
            st.caption(
                f"Gap creación → modificación (normalizado a UTC): {gap_seconds:.0f} segundos "
                "— calculado convirtiendo ambas fechas a UTC, no restando las horas locales crudas."
            )

        if any(p.get("looks_like_scanned_image") for p in text_layer.get("pages", [])):
            st.warning("Al menos una página parece ser una imagen escaneada sin capa de texto real.")

        st.markdown("---")
        st.markdown("#### Metadata estándar")
        st.write(f"**Producer:** {meta.get('producer') or 'No detectado'}")
        st.write(f"**Creator:** {meta.get('creator') or 'No detectado'}")
        st.write(f"**Author:** {meta.get('author') or 'No detectado'}")
        st.write(f"**Título:** {meta.get('title') or 'No detectado'}")
        creation = meta.get("creation_date") or {}
        mod = meta.get("mod_date") or {}
        st.write(f"**Fecha de creación:** {creation.get('raw') or 'No detectado'}")
        st.write(f"**Fecha de modificación:** {mod.get('raw') or 'No detectado'}")

        st.markdown("---")
        st.markdown("#### 📋 Resumen copiable (completo)")
        st.caption("Todos los campos extraídos, listos para pegar y analizar.")
        st.code(build_summary(uploaded_file.name, data), language=None)

    with tabs[1]:
        st.json(meta)

    with tabs[2]:
        st.json(structure)
        st.json(file_info)

    with tabs[3]:
        st.write(f"**¿Tiene capa de texto?:** {'Sí' if text_layer.get('has_text_layer') else 'No'}")
        st.write(f"**Total de caracteres extraídos:** {text_layer.get('total_char_count', 0)}")
        render_markdown_table(text_layer.get("pages", []))

    with tabs[4]:
        st.caption(signatures.get("note", ""))
        st.write(f"**sigflags:** {signatures.get('sigflags')}")
        st.write(f"**¿Tiene campos de firma?:** {'Sí' if signatures.get('has_signature_fields') else 'No'}")
        if signatures.get("fields"):
            render_markdown_table(signatures["fields"])
        else:
            st.info("No se encontraron campos de firma en el documento.")

    with tabs[5]:
        if embedded_files:
            render_markdown_table(embedded_files)
        else:
            st.info("El PDF no tiene archivos embebidos.")

    with tabs[6]:
        if xmp.get("present"):
            if xmp.get("truncated"):
                st.caption("XMP truncado a 8000 caracteres.")
            st.code(xmp.get("raw_xml", ""), language="xml")
        else:
            st.info("El PDF no tiene metadata XMP.")


def build_summary(file_name, data):
    """
    Every field the backend extracted, in one plain-text block meant to be pasted
    elsewhere for analysis — deliberately exhaustive rather than curated, since the
    point of this branch is figuring out what's even there before deciding what
    matters.
    """
    file_info = data.get("file", {})
    meta = data.get("standard_metadata", {})
    structure = data.get("structure", {})
    text_layer = data.get("text_layer", {})
    signatures = data.get("signatures", {})
    embedded_files = data.get("embedded_files", [])
    xmp = data.get("xmp_metadata", {})

    def fmt(value):
        return value if value not in (None, "") else "No detectado"

    lines = [f"METADATA PDF — {file_name}", "=" * 60]

    lines.append("\n[ARCHIVO]")
    lines.append(f"Tamaño: {file_info.get('size_bytes', 0)} bytes")
    lines.append(f"SHA256: {file_info.get('sha256')}")
    lines.append(f"Versión PDF: {fmt(file_info.get('pdf_version'))}")

    lines.append("\n[METADATA ESTÁNDAR]")
    lines.append(f"Title: {fmt(meta.get('title'))}")
    lines.append(f"Author: {fmt(meta.get('author'))}")
    lines.append(f"Subject: {fmt(meta.get('subject'))}")
    lines.append(f"Keywords: {fmt(meta.get('keywords'))}")
    lines.append(f"Producer: {fmt(meta.get('producer'))}")
    lines.append(f"Creator: {fmt(meta.get('creator'))}")
    lines.append(f"Trapped: {fmt(meta.get('trapped'))}")
    creation = meta.get("creation_date") or {}
    mod = meta.get("mod_date") or {}
    lines.append(f"Fecha de creación (raw): {fmt(creation.get('raw'))}")
    if creation.get("parsed"):
        lines.append(
            f"  -> {creation.get('year')}-{creation.get('month'):02d}-{creation.get('day'):02d} "
            f"{creation.get('hour') or 0:02d}:{creation.get('minute') or 0:02d}:{creation.get('second') or 0:02d} "
            f"TZ={fmt(creation.get('timezone'))}"
        )
    lines.append(f"Fecha de modificación (raw): {fmt(mod.get('raw'))}")
    if mod.get("parsed"):
        lines.append(
            f"  -> {mod.get('year')}-{mod.get('month'):02d}-{mod.get('day'):02d} "
            f"{mod.get('hour') or 0:02d}:{mod.get('minute') or 0:02d}:{mod.get('second') or 0:02d} "
            f"TZ={fmt(mod.get('timezone'))}"
        )
    gap_seconds = meta.get("creation_to_modification_gap_seconds")
    lines.append(
        f"Gap creación->modificación (normalizado a UTC): {gap_seconds:.0f}s"
        if gap_seconds is not None else "Gap creación->modificación: no calculable (falta TZ en alguna fecha)"
    )

    lines.append("\n[ESTRUCTURA / REVISIONES]")
    lines.append(f"Páginas: {structure.get('page_count')}")
    lines.append(f"Encriptado: {structure.get('is_encrypted')}")
    lines.append(f"Requiere contraseña: {structure.get('needs_password')}")
    lines.append(f"Permisos (bitmask): {fmt(structure.get('permissions'))}")
    lines.append(f"Cantidad de entradas xref: {structure.get('xref_entry_count')}")
    lines.append(f"Marcadores %%EOF: {structure.get('eof_marker_count')}")
    lines.append(f"Marcadores startxref: {structure.get('startxref_count')}")
    lines.append(f"Posible edición incremental: {structure.get('likely_incrementally_updated')}")

    lines.append("\n[CAPA DE TEXTO]")
    lines.append(f"Tiene capa de texto: {text_layer.get('has_text_layer')}")
    lines.append(f"Total de caracteres extraídos: {text_layer.get('total_char_count', 0)}")
    for page in text_layer.get("pages", []):
        lines.append(
            f"  Página {page.get('page')}: {page.get('char_count')} caracteres, "
            f"{page.get('image_count')} imágenes, "
            f"¿parece escaneada?: {page.get('looks_like_scanned_image')}"
        )

    lines.append("\n[FIRMAS DIGITALES]")
    lines.append(f"sigflags: {signatures.get('sigflags')}")
    lines.append(f"Tiene campos de firma: {signatures.get('has_signature_fields')}")
    if signatures.get("fields"):
        for field in signatures["fields"]:
            lines.append(f"  Página {field.get('page')}: campo '{field.get('field_name')}' — firmado: {field.get('is_signed')}")
    else:
        lines.append("  (sin campos de firma)")

    lines.append("\n[ARCHIVOS EMBEBIDOS]")
    if embedded_files:
        for ef in embedded_files:
            lines.append(f"  {ef.get('name')} ({ef.get('filename')}) — {ef.get('size_bytes')} bytes — {fmt(ef.get('description'))}")
    else:
        lines.append("  (ninguno)")

    lines.append("\n[XMP]")
    lines.append(f"Presente: {xmp.get('present')}")
    if xmp.get("present"):
        lines.append(f"Truncado: {xmp.get('truncated')}")
        lines.append("--- XML crudo ---")
        lines.append(xmp.get("raw_xml") or "")

    return "\n".join(lines)


if uploaded_file is not None:
    analyze_btn = st.button("Analizar PDF", width="stretch", type="primary")

    if analyze_btn:
        with st.spinner("Analizando..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(API_URL, files=files)

                if response.status_code != 200:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    st.stop()

                result = response.json()
                render_result(uploaded_file, result)

            except requests.exceptions.ConnectionError:
                st.error("ERROR: Could not connect to the Backend API.")
