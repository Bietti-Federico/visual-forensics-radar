import logging
import os

import numpy as np
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

    CORNER_DISTANCE_THRESHOLD = 60.0
    SATURATION_THRESHOLD = 40.0

    def _estimate_capture_mode(self, image: Image.Image) -> dict:
        """
        Distinguishes a photo of a physical document (paper on a table/tablecloth,
        background visible around it, natural lighting and a slight off-axis angle)
        from a flat digital source (a screenshot or rendered PDF, usually a uniform
        light background filling the frame edge-to-edge). Typography naturally has
        more baseline variance in a photo — paper curvature, shadows, perspective —
        so this lets typography relax its anomaly thresholds for physical photos
        instead of holding them to the same strictness as a flat digital render.
        """
        try:
            rgb = image.convert("RGB")
            width, height = rgb.size
            patch = max(10, min(width, height) // 12)

            corners = [
                rgb.crop((0, 0, patch, patch)),
                rgb.crop((width - patch, 0, width, patch)),
                rgb.crop((0, height - patch, patch, height)),
                rgb.crop((width - patch, height - patch, width, height)),
            ]
            corner_means = [
                np.asarray(corner, dtype=np.float64).reshape(-1, 3).mean(axis=0)
                for corner in corners
            ]
            # A scan/screenshot's corners are usually near-white; a photo's corners
            # show whatever surface the document was placed on.
            max_corner_distance = max(float(np.linalg.norm(mean - 255.0)) for mean in corner_means)

            saturation = np.asarray(rgb.convert("HSV"), dtype=np.float64)[:, :, 1]
            mean_saturation = float(saturation.mean())

            is_photo = (
                max_corner_distance > self.CORNER_DISTANCE_THRESHOLD
                or mean_saturation > self.SATURATION_THRESHOLD
            )

            return {
                "capture_mode": "photo" if is_photo else "digital_or_scan",
                "max_corner_distance_from_white": round(max_corner_distance, 1),
                "mean_saturation": round(mean_saturation, 1),
            }
        except Exception as e:
            logger.error(f"Error estimating capture mode: {str(e)}")
            return {"capture_mode": "digital_or_scan", "max_corner_distance_from_white": None, "mean_saturation": None}

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

            capture_mode_info = self._estimate_capture_mode(image)

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
                **capture_mode_info,
            }

            logger.info("Metadata analysis complete.")
            return result

        except Exception as e:
            logger.error(f"Error during metadata analysis: {str(e)}")
            return {"status": "error", "message": str(e)}