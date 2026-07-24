import logging
import os
from PIL import Image, ExifTags

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetadataDetector")


class MetadataDetector:
    """
    Lightweight metadata inspector for uploaded images.
    Extracts EXIF and file-level properties useful for fast triage.
    """

    EDITING_SOFTWARE_KEYWORDS = (
        "photoshop",
        "lightroom",
        "gimp",
        "canva",
        "pixlr",
        "photopea",
        "snapseed",
        "affinity",
    )

    def analyze(self, image_path: str) -> dict:
        try:
            file_size = os.path.getsize(image_path)
            image = Image.open(image_path)

            exif = {}
            try:
                raw_exif = image.getexif()
                exif = {
                    ExifTags.TAGS.get(tag_id, str(tag_id)): value
                    for tag_id, value in raw_exif.items()
                }
            except Exception:
                exif = {}

            software = str(exif.get("Software", "")).strip()
            make = str(exif.get("Make", "")).strip()
            model = str(exif.get("Model", "")).strip()
            datetime_original = str(exif.get("DateTimeOriginal", "")).strip()
            dpi = exif.get("XResolution") or exif.get("YResolution")

            suspicious_signals = []
            score = 0

            if not exif:
                score += 1
                suspicious_signals.append("La imagen no contiene metadatos EXIF útiles.")

            if software:
                lowered = software.lower()
                if any(keyword in lowered for keyword in self.EDITING_SOFTWARE_KEYWORDS):
                    score += 2
                    suspicious_signals.append(f"El campo Software indica edición: {software}.")
                else:
                    suspicious_signals.append(f"Software registrado: {software}.")

            if make or model:
                suspicious_signals.append(f"Dispositivo registrado: {make} {model}".strip())

            if datetime_original:
                suspicious_signals.append(f"Fecha EXIF original: {datetime_original}.")

            if dpi is None:
                score += 1
                suspicious_signals.append("No se encontró resolución EXIF (DPI).")

            if file_size < 20_000:
                score += 1
                suspicious_signals.append("Archivo muy pequeño para un recibo habitual.")

            if image.width < 500 or image.height < 500:
                score += 1
                suspicious_signals.append(f"Resolución baja: {image.width}x{image.height}.")

            if image.format not in {"JPEG", "PNG", "WEBP"}:
                score += 1
                suspicious_signals.append(f"Formato poco habitual para un recibo: {image.format}.")

            result = {
                "status": "success",
                "file_size_bytes": file_size,
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "has_exif": bool(exif),
                "software": software,
                "make": make,
                "model": model,
                "datetime_original": datetime_original,
                "suspicious_signals": suspicious_signals,
                "metadata_score": min(score, 4),
            }

            logger.info("Metadata analysis complete.")
            return result

        except Exception as e:
            logger.error(f"Error during metadata analysis: {str(e)}")
            return {"status": "error", "message": str(e)}