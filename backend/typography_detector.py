import logging
import re

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
    MIN_COMPONENT_PIXELS = 5
    MIN_RELIABLE_SHAPE_SAMPLES = 3

    # A well-formed Argentine peso amount: optional "$", digits grouped by thousands
    # with ".", exactly 2 decimal digits after ",". Table columns that aren't monetary
    # amounts (quantities, percentages, day counts — e.g. "22.00" units) often still
    # match the OCR-level looser `is_amount` pattern, but rarely this stricter shape;
    # requiring it here keeps the "amounts" bucket to fields actually rendered as money.
    WELL_FORMED_AMOUNT_PATTERN = re.compile(r"^\$?\s*\d{1,3}(\.\d{3})*,\d{2}$")

    def _bucket_for(self, region: dict) -> str | None:
        """
        Only true currency/date fields are compared against each other. Generic numeric
        strings (account numbers, CUIL/CUIT, reference codes, quantity/unit columns)
        are excluded from the "amounts" bucket even when `is_key_field` — they aren't
        rendered the same way as monetary amounts, so mixing them in would contaminate
        the population with unrelated glyph shapes and trigger false positives.
        """
        if region.get("is_date"):
            return "dates"
        if region.get("is_amount") and self.WELL_FORMED_AMOUNT_PATTERN.match((region.get("text") or "").strip()):
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

        slant_angle, aspect_ratio, shape_reliable = self._shape_features(ink_mask)

        return {
            "height_px": float(y2 - y1),
            "ink_ratio": ink_ratio,
            "mean_ink_intensity": mean_ink_intensity,
            "ink_std": ink_std,
            "slant_angle": slant_angle,
            "aspect_ratio": aspect_ratio,
            "shape_reliable": shape_reliable,
        }

    def _connected_components(self, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Minimal 8-connectivity flood-fill labeling (no scipy dependency, crops are tiny
        so a plain Python BFS is fast enough) to segment individual glyphs within the
        ink mask. Needed because a multi-digit amount's overall ink bounding box is
        dominated by how many digits it has, not by the shape of any single letter —
        measuring the whole crop as one blob confounds string length with font shape.
        """
        visited = np.zeros_like(mask, dtype=bool)
        height, width = mask.shape
        neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        components = []

        for start_y in range(height):
            for start_x in range(width):
                if not mask[start_y, start_x] or visited[start_y, start_x]:
                    continue

                stack = [(start_y, start_x)]
                visited[start_y, start_x] = True
                ys_list, xs_list = [], []

                while stack:
                    y, x = stack.pop()
                    ys_list.append(y)
                    xs_list.append(x)
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                components.append((np.array(ys_list), np.array(xs_list)))

        return components

    def _component_shape(self, ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect_ratio = width / height if height > 0 else 1.0

        cx, cy = xs.mean(), ys.mean()
        dx, dy = xs - cx, ys - cy
        mu20 = float(np.mean(dx * dx))
        mu02 = float(np.mean(dy * dy))
        mu11 = float(np.mean(dx * dy))

        if abs(mu20 - mu02) < 1e-9 and abs(mu11) < 1e-9:
            slant_angle = 0.0
        else:
            slant_angle = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))

        return slant_angle, aspect_ratio

    def _shape_features(self, ink_mask: np.ndarray) -> tuple[float, float, bool]:
        """
        Estimates font/handwriting shape PER CHARACTER (each connected ink component),
        then takes the median across characters found in the crop:
        - slant_angle: dominant stroke angle (degrees) from second-order image moments
          of a single glyph — a font swap or handwritten digit typically has a
          different slant than the printed/rendered text around it.
        - aspect_ratio: width/height of a single glyph's tight ink bounding box —
          captures letter proportions, independent of how many characters the field has.
        Falls back to neutral defaults (0.0 slant, 1.0 aspect, reliable=False) when no
        component has enough ink to measure confidently — callers must exclude
        unreliable fields from the comparison population instead of treating the
        neutral default as a real measurement.
        """
        components = self._connected_components(ink_mask)
        shapes = [
            self._component_shape(ys, xs)
            for ys, xs in components
            if ys.size >= self.MIN_COMPONENT_PIXELS
        ]
        if not shapes:
            return 0.0, 1.0, False

        slants = [slant for slant, _ in shapes]
        aspects = [aspect for _, aspect in shapes]
        return float(np.median(slants)), float(np.median(aspects)), True

    def _leave_one_out_z_score(self, values: list[float], index: int, mad_floor: float = 0.0) -> float:
        """
        Modified (Iglewicz-Hoaglin) z-score of values[index] against the median/MAD
        of every OTHER value in the same bucket, so a field is never compared
        against a population that includes itself. `mad_floor` sets the smallest
        deviation still treated as meaningful — without it, if more than half the
        bucket ties on the same value (common for e.g. `aspect_ratio` when several
        fields fall back to the neutral 1.0 default), MAD collapses to 0 and every
        z-score in the bucket goes silently to 0.0, even for a field that's genuinely
        a bit off from the rest.
        """
        others = values[:index] + values[index + 1:]
        if len(others) < 2:
            return 0.0

        median = float(np.median(others))
        mad = max(float(np.median(np.abs(np.array(others) - median))), mad_floor)
        if mad == 0:
            return 0.0

        return abs(0.6745 * (values[index] - median) / mad)

    def _masked_leave_one_out_z_score(
        self, values: list[float], reliable_mask: list[bool], index: int, mad_floor: float = 0.0
    ) -> float:
        """
        Same as `_leave_one_out_z_score`, but the comparison population is restricted to
        entries where `reliable_mask` is True. slant_angle/aspect_ratio fall back to a
        neutral placeholder (not a real measurement) when a crop has too little ink —
        comparing a real value against a population padded with placeholders (or an
        unreliable value against anything) produces a z-score that looks confident but
        is statistically meaningless, so both cases return 0.0 instead.
        """
        if not reliable_mask[index]:
            return 0.0

        others = [v for i, v in enumerate(values) if i != index and reliable_mask[i]]
        if len(others) < self.MIN_RELIABLE_SHAPE_SAMPLES - 1:
            return 0.0

        median = float(np.median(others))
        mad = max(float(np.median(np.abs(np.array(others) - median))), mad_floor)
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

            feature_series = {
                "height": [entry["features"]["height_px"] for entry in entries],
                "ink_ratio": [entry["features"]["ink_ratio"] for entry in entries],
            }
            # slant_angle/aspect_ratio fall back to a neutral placeholder when a crop
            # lacks enough ink to measure — those entries must not be compared as if
            # they were real values, in either direction (as the entry being judged, or
            # as part of the reference population for another entry).
            shape_reliable = [entry["features"]["shape_reliable"] for entry in entries]
            masked_feature_series = {
                "slant_angle": [entry["features"]["slant_angle"] for entry in entries],
                "aspect_ratio": [entry["features"]["aspect_ratio"] for entry in entries],
            }
            # Smallest deviation still treated as meaningful, in each feature's own scale.
            mad_floors = {
                "height": 1.0,
                "ink_ratio": 0.02,
                "slant_angle": 1.0,
                "aspect_ratio": 0.05,
            }

            for idx, entry in enumerate(entries):
                z_scores = {
                    name: self._leave_one_out_z_score(series, idx, mad_floor=mad_floors[name])
                    for name, series in feature_series.items()
                }
                z_scores.update({
                    name: self._masked_leave_one_out_z_score(series, shape_reliable, idx, mad_floor=mad_floors[name])
                    for name, series in masked_feature_series.items()
                })
                dominant_feature = max(z_scores, key=z_scores.get)
                max_abs_z = z_scores[dominant_feature]

                entry["z_scores"] = {name: round(z, 2) for name, z in z_scores.items()}
                entry["max_abs_z"] = round(max_abs_z, 2)
                entry["dominant_feature"] = dominant_feature
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
