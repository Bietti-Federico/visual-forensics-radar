import logging

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TypographyDetector")


def _otsu_threshold(pixels: np.ndarray) -> tuple[float, float]:
    """
    Returns (threshold, between_class_variance) so callers can gauge how reliable the
    split actually is (a near-blank or otherwise low-contrast crop has a low between-class
    variance and shouldn't be trusted).
    """
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

    # Normalize into roughly [0, 1]: 1.0 would be a perfect 50/50 split spanning the
    # full 0-255 range, so the result is comparable across crops of different sizes.
    max_possible_variance = (total ** 2) * (255.0 ** 2) / 4.0
    separability = best_variance / max_possible_variance if max_possible_variance > 0 and best_variance > 0 else 0.0
    return best_threshold, separability


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
    MIN_SEPARABILITY = 0.03

    def _bucket_for(self, region: dict) -> str | None:
        """
        Only true currency/date fields are compared against each other. Generic numeric
        strings (account numbers, CUIL/CUIT, reference codes) are excluded from the
        "amounts" bucket even when `is_key_field` — they aren't rendered the same way as
        monetary amounts, so mixing them in would contaminate the population with
        unrelated glyph shapes and trigger false positives.
        """
        if region.get("is_date"):
            return "dates"
        if region.get("is_amount"):
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

        threshold, separability = _otsu_threshold(pixels.astype(np.uint8))
        if separability < self.MIN_SEPARABILITY:
            # No reliable bimodal split (e.g. a near-blank cell, or a low-contrast crop) —
            # any feature computed here would be noise, so skip this field entirely
            # instead of feeding a garbage sample into the bucket's statistics.
            return None

        below_mask = pixels < threshold
        below_ratio = float(below_mask.mean())
        # Text strokes are the minority of pixels in a tight crop, whether the field is
        # dark-on-light (a printed receipt) or light-on-dark (a highlighted "Total" row
        # in a homebanking screenshot) — always treat the smaller side as "ink" instead
        # of assuming dark-on-light, which would otherwise flag every inverted-contrast
        # UI row as a typography anomaly.
        if below_ratio <= 0.5:
            ink_mask = below_mask
            ink_ratio = below_ratio
        else:
            ink_mask = ~below_mask
            ink_ratio = 1.0 - below_ratio

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
