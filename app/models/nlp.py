from transformers import (
    Pipeline,
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Any

class STARAnalyzer:
    def __init__(self):
        # Load models
        # Using BERT model fine-tuned for text classification
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        
        # Sentence transformer for semantic similarity and coherence
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Zero-shot classification pipeline for STAR components
        self.zero_shot = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli")

    def analyze_star_components(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to identify STAR components using zero-shot classification.
        """
        # Define STAR components and their descriptions
        star_components = {
            "situation": "description of the context or background of an event",
            "task": "explanation of the responsibility or challenge that needed to be addressed",
            "action": "specific steps taken to handle the situation",
            "result": "outcome, impact, or lessons learned from the actions taken"
        }
        
        # Split text into sentences for analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        results = {}
        for component, description in star_components.items():
            # Classify each sentence against current STAR component
            component_scores = self.zero_shot(
                sentences,
                candidate_labels=[component, f"not_{component}"],
                hypothesis_template="This sentence describes a {}"
            )
            
            # Find sentences that match this component
            matching_sentences = []
            for sent, scores in zip(sentences, component_scores):
                if scores['labels'][0] == component and scores['scores'][0] > 0.7:
                    matching_sentences.append(sent)
            
            results[component] = {
                "found": len(matching_sentences) > 0,
                "text": " ".join(matching_sentences) if matching_sentences else "",
                "confidence": max([s['scores'][0] for s in component_scores]) if component_scores else 0
            }
        
        return results

    def analyze_coherence(self, text: str) -> float:
        """
        Analyze the coherence of the response using sentence embeddings.
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0
            
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(sentences)
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(similarity)
            
        # Return average similarity as coherence score
        return float(np.mean(similarities))

    def analyze_response(self, text: str) -> Dict[str, Any]:
        """
        Perform complete analysis of an interview response.
        """
        # Get STAR component analysis
        star_analysis = self.analyze_star_components(text)
        
        # Get coherence score
        coherence_score = self.analyze_coherence(text)
        
        # Combine results
        analysis = {
            "star_components": star_analysis,
            "coherence_score": coherence_score,
            "overall_score": self._calculate_overall_score(star_analysis, coherence_score)
        }
        
        # Add feedback
        analysis["feedback"] = self._generate_feedback(analysis)
        
        return analysis

    def _calculate_overall_score(self, star_analysis: Dict, coherence_score: float) -> float:
        """
        Calculate overall response score based on STAR components and coherence.
        """
        # Weight for each component
        star_weight = 0.7  # STAR components
        coherence_weight = 0.3  # Coherence score
        
        # Calculate STAR score
        star_score = sum(
            comp["confidence"] for comp in star_analysis.values()
        ) / len(star_analysis)
        
        # Combine scores
        return (star_score * star_weight) + (coherence_score * coherence_weight)

    def _generate_feedback(self, analysis: Dict) -> Dict[str, str]:
        """
        Generate specific feedback based on the analysis.
        """
        feedback = {}
        star_components = analysis["star_components"]
        
        # Feedback for each STAR component
        for component, data in star_components.items():
            if not data["found"]:
                feedback[component] = f"Your response is missing a clear {component}. Try to include..."
            elif data["confidence"] < 0.8:
                feedback[component] = f"The {component} could be stronger. Consider..."
            else:
                feedback[component] = f"Good job describing the {component}!"
                
        # Coherence feedback
        if analysis["coherence_score"] < 0.7:
            feedback["coherence"] = "Try to make your response more coherent by..."
        else:
            feedback["coherence"] = "Your response flows well!"
            
        return feedback