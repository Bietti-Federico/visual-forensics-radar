import logging

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TypographyDetector")


def _otsu_threshold(pixels: np.ndarray) -> float:
    histogram, _ = np.histogram(pixels, bins=256, range=(0, 256))
    total = int(pixels.size)
    sum_total = float(np.dot(np.arange(256), histogram))

    sum_background = 0.0
    weight_background = 0
    best_threshold = 128.0
    best_variance = -1.0

    for threshold in range(256):
        weight_background += int(histogram[threshold])
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = float(threshold)

    return best_threshold


class TypographyDetector:
    """
    Lightweight font/print-consistency check for key numeric and date fields.
    Compares simple rendering features (glyph height, ink density) of each field
    crop against the other comparable fields in the same document, to flag a
    field whose typography stands out from the rest of the receipt — a common
    signature of a pasted-in or hand-edited value that survives a print/rescan
    cycle and would otherwise be invisible to compression-based ELA analysis.
    """

    MIN_SAMPLES_PER_BUCKET = {"amounts": 5, "dates": 3}
    Z_SCORE_THRESHOLD = 3.5
    MIN_CROP_SIZE = 6
    CROP_PADDING = 3

    def _bucket_for(self, region: dict) -> str | None:
        if region.get("is_date"):
            return "dates"
        if region.get("is_numeric") or region.get("is_key_field"):
            return "amounts"
        return None

    def _extract_features(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> dict | None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - self.CROP_PADDING)
        y1 = max(0, y1 - self.CROP_PADDING)
        x2 = min(image.width, x2 + self.CROP_PADDING)
        y2 = min(image.height, y2 + self.CROP_PADDING)

        if (x2 - x1) < self.MIN_CROP_SIZE or (y2 - y1) < self.MIN_CROP_SIZE:
            return None

        crop = image.crop((x1, y1, x2, y2))
        pixels = np.asarray(crop, dtype=np.float64)
        if pixels.size == 0:
            return None

        threshold = _otsu_threshold(pixels.astype(np.uint8))
        ink_mask = pixels < threshold
        ink_ratio = float(ink_mask.mean())

        if ink_mask.any():
            ink_pixels = pixels[ink_mask]
            mean_ink_intensity = float(ink_pixels.mean())
            ink_std = float(ink_pixels.std())
        else:
            mean_ink_intensity = float(pixels.mean())
            ink_std = float(pixels.std())

        return {
            "height_px": float(y2 - y1),
            "ink_ratio": ink_ratio,
            "mean_ink_intensity": mean_ink_intensity,
            "ink_std": ink_std,
        }

    def _leave_one_out_z_score(self, values: list[float], index: int) -> float:
        """
        Modified (Iglewicz-Hoaglin) z-score of values[index] against the median/MAD
        of every OTHER value in the same bucket, so a field is never compared
        against a population that includes itself.
        """
        others = values[:index] + values[index + 1:]
        if len(others) < 2:
            return 0.0

        median = float(np.median(others))
        mad = float(np.median(np.abs(np.array(others) - median)))
        if mad == 0:
            return 0.0

        return abs(0.6745 * (values[index] - median) / mad)

    def analyze(self, image_source: str | Image.Image, candidate_regions: list[dict]) -> dict:
        try:
            if isinstance(image_source, Image.Image):
                image = image_source.convert("L")
            else:
                image = Image.open(image_source).convert("L")
        except Exception as e:
            logger.error(f"Error opening image for typography analysis: {str(e)}")
            return {"status": "error", "message": str(e)}

        buckets: dict[str, list[dict]] = {"amounts": [], "dates": []}

        for region in candidate_regions:
            bucket = self._bucket_for(region)
            if bucket is None or not region.get("bbox"):
                continue

            features = self._extract_features(image, tuple(region["bbox"]))
            if features is None:
                continue

            buckets[bucket].append({
                "text": region.get("text", ""),
                "bbox": region.get("bbox"),
                "is_key_field": bool(region.get("is_key_field")),
                "features": features,
            })

        bucket_summaries = {}
        anomalous_fields = []

        for bucket_name, entries in buckets.items():
            min_samples = self.MIN_SAMPLES_PER_BUCKET.get(bucket_name, 4)
            if len(entries) < min_samples:
                bucket_summaries[bucket_name] = {
                    "status": "insufficient_population",
                    "sample_count": len(entries),
                }
                continue

            heights = [entry["features"]["height_px"] for entry in entries]
            ink_ratios = [entry["features"]["ink_ratio"] for entry in entries]

            for idx, entry in enumerate(entries):
                height_z = self._leave_one_out_z_score(heights, idx)
                ink_z = self._leave_one_out_z_score(ink_ratios, idx)
                max_abs_z = max(height_z, ink_z)

                entry["max_abs_z"] = round(max_abs_z, 2)
                entry["typography_anomaly"] = max_abs_z > self.Z_SCORE_THRESHOLD
                entry["bucket"] = bucket_name

                if entry["typography_anomaly"]:
                    anomalous_fields.append(entry)

            bucket_summaries[bucket_name] = {
                "status": "success",
                "sample_count": len(entries),
            }

        anomalous_fields.sort(key=lambda entry: entry["max_abs_z"], reverse=True)

        result = {
            "status": "success",
            "buckets": bucket_summaries,
            "anomalous_fields": anomalous_fields,
        }
        logger.info(f"Typography analysis complete. {len(anomalous_fields)} anomalous field(s) found.")
        return result
