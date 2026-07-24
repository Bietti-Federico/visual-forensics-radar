import os
import logging
import tempfile
from PIL import Image, ImageChops, ImageEnhance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger("ElaDetector")

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

    def _analyze_image(self, original_img: Image.Image, output_path: str | None = "ela_heatmap.jpg", anomaly_threshold: int | None = None) -> dict:
        """
        Analyze an in-memory image and optionally saves a heatmap.
        """
        temp_filename = None

        try:
            original_img = original_img.convert("RGB")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_filename = tmp_file.name

            original_img.save(temp_filename, "JPEG", quality=self.quality)
            compressed_img = Image.open(temp_filename).convert("RGB")

            # Calculate the mathematical difference between the two images (A - B)
            ela_image = ImageChops.difference(original_img, compressed_img)

            # Enhance brightness to make the pixel differences visible to the human eye
            # Find the maximum pixel difference to adjust the black/white balance scale

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])

            if max_diff == 0:
                max_diff = 1 # Prevent division by zero error on completely flat images
            
            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            if output_path:
                ela_image.save(output_path)

            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)

            if output_path:
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
                "threshold_used": threshold


            }
        
        except Exception as e:

            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            logger.error(f"Error during ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}

    def analyze(self, image_path:str, output_path:str="ela_heatmap.jpg", anomaly_threshold: int | None = None) -> dict:
        """
        Analyze an image from disk and optionally save a heatmap.
        """
        try:
            original_img = Image.open(image_path).convert("RGB")
            result = self._analyze_image(original_img, output_path=output_path, anomaly_threshold=anomaly_threshold)
            result["original_image"] = image_path
            return result
        except Exception as e:
            logger.error(f"Error during ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}

    def analyze_crop(self, image_path: str, crop_box: tuple[int, int, int, int], anomaly_threshold: int | None = None) -> dict:
        """
        Analyze a cropped region without generating a heatmap file.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            padded_box = crop_box
            crop = image.crop(padded_box)
            result = self._analyze_image(crop, output_path=None, anomaly_threshold=anomaly_threshold)
            result["crop_box"] = padded_box
            return result
        except Exception as e:
            logger.error(f"Error during cropped ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}
        

if __name__ == "__main__":

    detector = ElaDetector()

    test_image_path = "C:\\Users\\computer\\Desktop\\projects\\fake_detector\\images\\test2.png"
    output_heatmap_path = "C:\\Users\\computer\\Desktop\\projects\\fake_detector\\images\\_test2_ela_heatmap.jpg"

    result = detector.analyze(test_image_path,output_heatmap_path)
    print(result)


