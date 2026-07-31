import io
import logging

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger("ElaDetector")


def _ink_mask(pixels: np.ndarray) -> np.ndarray | None:
    """
    Lightweight, polarity-invariant Otsu ink mask (same approach as
    typography_detector.py) — picks out just the text/ink pixels within a crop,
    regardless of whether the field is dark-on-light or light-on-dark (a highlighted
    "Total" row with a colored fill and white text). Returns None if the crop has no
    reliable bimodal split to threshold on (e.g. a near-blank cell).
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

        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = float(threshold)

    max_possible_variance = (total ** 2) * (255.0 ** 2) / 4.0
    separability = best_variance / max_possible_variance if max_possible_variance > 0 and best_variance > 0 else 0.0
    if separability < 0.03:
        return None

    below = pixels < best_threshold
    below_ratio = float(below.mean())
    return below if below_ratio <= 0.5 else ~below


class ElaDetector:

    """
    Error Level Analysis (ELA) module
    Detects compression anomalies in an image and generates a heatmap
    highlighting potentially manipulated regions.
    Requires no AI/models; performs mathematical matrix operations directly on the CPU.
    """

    def __init__(self, quality:int = 90, anomaly_threshold: int = 35):
        self.quality = quality
        self.anomaly_threshold = anomaly_threshold
        logger.info(f"ELADetector initialized. Target Compression Quality: {self.quality}%, Threshold: {self.anomaly_threshold}")

    def _analyze_image(
        self,
        original_img: Image.Image,
        output_path: str | None = "ela_heatmap.jpg",
        anomaly_threshold: int | None = None,
        mask_to_ink: bool = False,
    ) -> dict:
        """
        Analyze an in-memory image and optionally saves a heatmap. When `mask_to_ink`
        is set, `max_difference` is measured only over the crop's own text/ink pixels
        instead of the whole crop — a highlighted field (a bold "Total" row with a
        colored fill) has a large uniform background block whose edges recompress with
        extra error on their own, unrelated to whether the text inside was edited;
        masking to just the ink sidesteps that instead of losing the field entirely.
        """
        converted_img = None
        compressed_img = None
        ela_image = None
        enhanced_image = None

        try:
            # Luminance only for the actual diff math — JPEG subsamples its color
            # (Cb/Cr) channels at lower resolution than luminance, so any color edge
            # (a logo, a colored header, a highlighted row) recompresses with extra
            # error that has nothing to do with tampering. Working in "L" bypasses
            # chroma subsampling entirely instead of picking it up as false signal.
            # This only affects ELA's own math, not the image callers pass in/reuse.
            converted_img = original_img.convert("L")

            # In-memory JPEG round-trip instead of a real temp file — this runs once
            # per candidate field (dozens of times per request), and writing/reading/
            # deleting an actual file that many times per request, relying on the
            # garbage collector to eventually close the PIL Image/file handles
            # instead of closing them deterministically, left native decoder state
            # piling up across requests (see .close() calls below).
            buffer = io.BytesIO()
            converted_img.save(buffer, "JPEG", quality=self.quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            compressed_img.load()  # force full decode now, while the buffer is valid

            # Calculate the mathematical difference between the two images (A - B)
            ela_image = ImageChops.difference(converted_img, compressed_img)

            # Enhance brightness to make the pixel differences visible to the human eye
            # Find the maximum pixel difference to adjust the black/white balance scale

            ink_mask = None
            if mask_to_ink:
                ink_mask = _ink_mask(np.asarray(converted_img, dtype=np.float64))

            if ink_mask is not None and ink_mask.any():
                diff_array = np.asarray(ela_image, dtype=np.float64)
                max_diff = int(diff_array[ink_mask].max())
            else:
                extrema = ela_image.getextrema()
                # "L" mode images give a single (min, max) tuple; RGB gives one per
                # channel. Handled generically in case a caller ever passes a non-"L"
                # image through.
                max_diff = max(channel[1] for channel in extrema) if isinstance(extrema[0], tuple) else extrema[1]

            if max_diff == 0:
                max_diff = 1 # Prevent division by zero error on completely flat images

            scale = 255.0 / max_diff
            enhanced_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            if output_path:
                enhanced_image.save(output_path)
                logger.info(f"ELA Analysis complete. Heatmap saved to: {output_path}")
            else:
                logger.info("ELA Analysis complete.")

            threshold = self.anomaly_threshold if anomaly_threshold is None else anomaly_threshold
            is_anomaly = bool(max_diff > threshold)

            return{

                "status":"success",
                "ela_heatmap_path":output_path,
                "max_difference":max_diff,
                "anomaly_detected": is_anomaly,
                "threshold_used": threshold,
                "ink_masked": bool(ink_mask is not None and ink_mask.any()),

            }

        except Exception as e:
            logger.error(f"Error during ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}
        finally:
            # Close every intermediate image WE created explicitly instead of
            # leaving it to the garbage collector — never close `original_img`
            # itself, since callers (e.g. main.py's per-field ELA loop) reuse that
            # same shared, already-open image across dozens of crops per request.
            for img in (compressed_img, ela_image, enhanced_image, converted_img):
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass

    @staticmethod
    def _load(image_source: str | Image.Image) -> Image.Image:
        """
        Accepts either a path or an already-opened PIL Image so callers doing many
        crops on the same photo (e.g. per-field local ELA) can decode it once and
        reuse it, instead of re-reading and re-decoding the full-resolution file
        from disk on every call.
        """
        if isinstance(image_source, Image.Image):
            # Image.convert() returns a full copy even when the mode already matches —
            # with a shared photo reused across 12 crops per request, that's a full-size
            # in-memory copy 12 times over for nothing. crop() further down never mutates
            # the source, so it's safe to hand back the same object untouched.
            return image_source if image_source.mode == "RGB" else image_source.convert("RGB")
        return Image.open(image_source).convert("RGB")

    def analyze(self, image_source: str | Image.Image, output_path: str | None = None, anomaly_threshold: int | None = None) -> dict:
        """
        Analyze an image (from disk or an already-opened PIL Image) and optionally save a heatmap.
        The heatmap is only written when output_path is explicitly provided.
        """
        try:
            original_img = self._load(image_source)
            result = self._analyze_image(original_img, output_path=output_path, anomaly_threshold=anomaly_threshold)
            result["original_image"] = image_source if isinstance(image_source, str) else "<in-memory>"
            return result
        except Exception as e:
            logger.error(f"Error during ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}

    def analyze_crop(
        self,
        image_source: str | Image.Image,
        crop_box: tuple[int, int, int, int],
        anomaly_threshold: int | None = None,
        mask_to_ink: bool = False,
    ) -> dict:
        """
        Analyze a cropped region without generating a heatmap file. `mask_to_ink=True`
        restricts the measurement to the crop's own text/ink pixels — use it for a
        field with a highlighted/colored background instead of skipping it outright.
        """
        crop = None
        try:
            image = self._load(image_source)
            padded_box = crop_box
            crop = image.crop(padded_box)
            result = self._analyze_image(crop, output_path=None, anomaly_threshold=anomaly_threshold, mask_to_ink=mask_to_ink)
            result["crop_box"] = padded_box
            return result
        except Exception as e:
            logger.error(f"Error during cropped ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}
        finally:
            # `crop` is a fresh, unshared Image made for this one call (unlike
            # `image`/`image_source`, which the caller may reuse across dozens of
            # other crops in the same request) — closing it explicitly here matters
            # for the same reason as _analyze_image's own cleanup above.
            if crop is not None:
                try:
                    crop.close()
                except Exception:
                    pass
        

if __name__ == "__main__":

    detector = ElaDetector()

    test_image_path = "C:\\Users\\computer\\Desktop\\projects\\fake_detector\\images\\test2.png"
    output_heatmap_path = "C:\\Users\\computer\\Desktop\\projects\\fake_detector\\images\\_test2_ela_heatmap.jpg"

    result = detector.analyze(test_image_path,output_heatmap_path)
    print(result)


