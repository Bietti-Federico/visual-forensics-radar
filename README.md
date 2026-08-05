# PDF Metadata Inspector

Rama de investigación: la ingesta se restringe a **PDFs descargados directamente de
la página oficial del banco/organismo** (sin fotos, sin capturas de pantalla). Antes
de decidir qué controles de fraude son viables sobre ese flujo, esta rama solo
extrae y muestra **todo lo que el archivo PDF en sí mismo revela**:

- Metadata estándar (Producer, Creator, Author, fechas de creación/modificación).
- Estructura del archivo: versión de PDF, cantidad de páginas, y señales de
  guardado/edición incremental posterior a la creación (marcadores `%%EOF`).
- Si el PDF tiene una capa de texto real o es una imagen escaneada sin texto
  extraíble, página por página.
- Presencia de campos de firma digital (sin verificación criptográfica del
  certificado).
- Archivos embebidos dentro del PDF.
- Metadata XMP cruda, si existe.

No hace OCR ni emite un veredicto de fraude — es una capa exploratoria para decidir,
banco por banco, qué es controlable.

## Estructura

```
├── backend/
│   └── pdf_metadata_extractor.py   # Extracción de metadata/estructura/firmas/XMP
├── frontend/
│   └── app.py                      # Streamlit Dashboard
├── main.py                         # FastAPI Entrypoint
├── Dockerfile
├── start.sh
└── requirements.txt
```

## Instalación local

```bash
docker build -t pdf-metadata-inspector .
docker run -p 8501:8501 -p 8000:8000 pdf-metadata-inspector
```

Abrir `http://localhost:8501`.
