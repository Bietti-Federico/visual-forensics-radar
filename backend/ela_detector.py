import os
import logging
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

    def analyze(self, image_path:str, output_path:str="ela_heatmap.jpg") -> dict:
        """
        Analyze the image and saves a heatmap that highlights manipualted pixels.
        """
        temp_filename = "temp_compressed.jpg"

        try:
            original_img = Image.open(image_path).convert("RGB")
            original_img.save(temp_filename,"JPEG",quality=self.quality)

            compressed_img = Image.open(temp_filename)

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

            ela_image.save(output_path)

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            logger.info(f"ELA Analysis complete. Heatmap saved to: {output_path}")

            is_anomaly = bool(max_diff > self.anomaly_threshold)

            return{

                "status":"success",
                "original_image":image_path,
                "ela_heatmap_path":output_path,
                "max_difference":max_diff,
                "anomaly_detected": is_anomaly


            }
        
        except Exception as e:

            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            logger.error(f"Error during ELA analysis: {str(e)}")
            return {"status":"error", "message":str(e)}
        

if __name__ == "__main__":

    detector = ElaDetector()

    test_image_path = "C:\\Users\\computer\\Desktop\\projects\\fake_detector\\images\\test2.png"
    output_heatmap_path = "C:\\Users\\computer\\Desktop\\projects\\fake_detector\\images\\_test2_ela_heatmap.jpg"

    result = detector.analyze(test_image_path,output_heatmap_path)
    print(result)


