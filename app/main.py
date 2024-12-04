from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.models.nlp import STARAnalyzer
from app.models.rule_based import RuleBasedSTARClassifier
from app.models.coherence import CoherenceAnalyzer
from typing import Dict, Any

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

star_classifier = RuleBasedSTARClassifier()
coherence_analyzer = CoherenceAnalyzer()

class ResponseAnalysisRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_response(request: AnalysisRequest) -> Dict[str, Any]:
    """
    Analyze an interview response using STAR method and coherence analysis
    """
    try:
        # Perform STAR analysis
        star_analysis = star_classifier.analyze(request.text)
        star_feedback = star_classifier.generate_feedback(star_analysis)
        
        # Perform coherence analysis
        coherence_analysis = coherence_analyzer.analyze(request.text)
        
        # Combine results
        return {
            "star_analysis": star_analysis,
            "star_feedback": star_feedback,
            "coherence_analysis": coherence_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoints for specific analyses
@app.post("/analyze/coherence")
async def analyze_coherence(request: ResponseAnalysisRequest) -> Dict[str, float]:
    """
    Analyze just the coherence of a response.
    """
    try:
        coherence_score = star_analyzer.analyze_coherence(request.text)
        return {"coherence_score": coherence_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/star")
async def analyze_star_components(request: ResponseAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze just the STAR components of a response.
    """
    try:
        star_analysis = star_analyzer.analyze_star_components(request.text)
        return {"star_components": star_analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)