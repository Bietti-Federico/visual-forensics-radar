import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.ela_detector import ElaDetector
from backend.clip_detector import ClipAuthenticator
from backend.vlm_explainer import VlmExplanier

logging.basicConfig(level=logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepFakeAPI")

# Global model dictionary (Ensures a single instance in VRAM)
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    SYSTEM ARCHITECTURE: Lifespan Management
    Loading models on every API request would instantly cause a VRAM overflow (OOM).
    Therefore, models are loaded into RAM/VRAM ONLY ONCE when the API starts.
    """

    logger.info("Loading AI Engines into VRAM... Please stand by.")
    try:
        models["ela"] = ElaDetector()
        models["clip"] = ClipAuthenticator()
        models["vlm"] = VlmExplanier()
        logger.info("All engines successfully initialized. API is ready to serve!")

    except Exception as e:
        logger.error(f"Critical error while loading models: {e}")

    yield
    
    logger.info("API is shutting down, clearing VRAM...")
    models.clear()

app = FastAPI(title="Deepfake Detection core API", version="1.0",lifespan=lifespan)

def decision_engine(ela_result: dict, clip_result:dict, vlm_result:dict) -> dict:
    """
    RULE-BASED EXPERT SYSTEM (Decision Engine)
    Evaluates outputs from 3 distinct AI engines and provides a weighted diagnosis.
    """

    ela_anomaly = ela_result.get("anomaly_detected",False)
    
    clip_probs = clip_result.get("detailed_probabilities",{})
    ai_prob = clip_probs.get("ai_probability",0)

    decision = {
        "title":"Diagnosis Inconclusive",
        "description":"Result are ambiguous",
        "risk_level":"Unknown"
    }

    # Pure AI Generation (e.g., Midjourney)
    if ai_prob > 75 and not ela_anomaly:
        decision["title"] = "Fully Synthetic (AI Generated)"
        decision["description"] = "The entire image is AI-generated (e.g., Midjourney/DALL-E). No partial manipulation detected; the image's core DNA is synthetic."
        decision["risk_level"] = "High (Red)"

    # Digital Forgery (e.g., Document Alteration / Photoshop)
    elif ai_prob < 40 and ela_anomaly:
        decision["title"] = "Partial Manipulation (Photoshop / Deepfake)"
        decision["description"] = "While the general composition belongs to a real photograph, a specific region has been digitally altered (pixel/frequency anomaly detected)."
        decision["risk_level"] = "High (Red)"

    # AI base + Human Edit
    elif ai_prob > 60 and ela_anomaly:
        decision["title"] = "Heavily Manipulated Synthetic"
        decision["description"] = "The image contains strong AI signatures and subsequent digital modifications. Reliability is zero."
        decision["risk_level"] = "Critical (Black)"

    # Authentic Image
    elif ai_prob < 40 and not ela_anomaly:
        decision["title"] = "Original / Clean Image"
        decision["description"] = "Our systems detected no traces of AI generation or subsequent digital manipulation. The image appears authentic."
        decision["risk_level"] = "Low (Green)"

    # Edge cases / Uncertain
    else:
        decision["title"] = "Suspicious Image"
        decision["description"] = f"The image carries partial synthetic traces (AI Score: {ai_prob:.1f}%). Reviewing the VLM Detective Report is highly recommended."
        decision["risk_level"] = "Medium (Yellow)"

    return decision

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    The main endpoint that receives the image from the frontend and orchestrates the 3 engines.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,detail="Please upload a valid image file.")
    
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir,file.filename)

    try:
        with open(temp_file_path,"wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"new analysis request received. Processing file: {file.filename}")

        logger.info("Step 1: Running ELA (Pixel) Analysis...")
        ela_res = models["ela"].analyze(temp_file_path)
            
        logger.info("Step 2: Running CLIP (Semantic) Analysis...")
        clip_res = models["clip"].analyze(temp_file_path)
            
        logger.info("Step 3: Running VLM (Logical) Analysis...")
        vlm_res = models["vlm"].analyze(temp_file_path)

        final_decision = decision_engine(ela_res, clip_res, vlm_res)

        response_data = {
            "status":"success",
            "final_decision": final_decision,
            "diagnostics":{
                "ela_layer":ela_res,
                "clip_layer":clip_res,
                "vlm_detective_report":vlm_res.get("explanation","")
            }
        }

        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")
    
    # Garbage collection
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file deleted securely: {file.filename}")