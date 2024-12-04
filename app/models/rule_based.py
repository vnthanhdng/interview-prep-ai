import spacy
from typing import Dict, List, Tuple
import re

class RuleBasedSTARClassifier:
    def __init__(self):
        # Load spaCy model for advanced text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Keywords and phrases for each STAR component
        self.keywords = {
            "situation": [
                "when", "while", "during", "at the time", "in my role",
                "previously", "recently", "last year", "in my previous",
                "faced", "encountered", "experienced"
            ],
            "task": [
                "needed to", "had to", "responsible for", "assigned to",
                "required to", "tasked with", "goal was", "objective",
                "challenge was", "project required"
            ],
            "action": [
                "i implemented", "i created", "i developed", "i coordinated",
                "i led", "i organized", "i managed", "i initiated",
                "i designed", "took steps", "decided to"
            ],
            "result": [
                "resulted in", "accomplished", "achieved", "improved",
                "increased", "decreased", "reduced", "succeeded",
                "led to", "outcome", "impact", "consequently"
            ]
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into sentences with spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def _analyze_sentence(self, sentence: str) -> List[Tuple[str, float]]:
        """Analyze a sentence for STAR components and return components with confidence scores"""
        sentence = sentence.lower()
        components = []
        
        # Check each component
        for component, phrases in self.keywords.items():
            confidence = 0
            matches = 0
            
            for phrase in phrases:
                if phrase in sentence:
                    matches += 1
                    
            if matches > 0:
                # Calculate confidence based on number of matching phrases
                confidence = min(matches * 0.3, 0.9)  # Cap at 0.9
                components.append((component, confidence))
                
        return components

    def _analyze_verb_tense(self, sentence: str) -> Dict[str, float]:
        """Analyze verb tense to help identify STAR components"""
        doc = self.nlp(sentence)
        
        # Count different verb tenses
        past_tense = 0
        present_tense = 0
        future_tense = 0
        
        for token in doc:
            if token.pos_ == "VERB":
                if "Past" in token.morph.get("Tense", []):
                    past_tense += 1
                elif "Pres" in token.morph.get("Tense", []):
                    present_tense += 1
                
        return {
            "situation": past_tense * 0.2,  # Situations often use past tense
            "action": past_tense * 0.3,     # Actions usually in past tense
            "result": present_tense * 0.2   # Results might use present tense
        }

    def analyze(self, text: str) -> Dict[str, List[Dict[str, any]]]:
        """
        Analyze text and identify STAR components
        Returns a dictionary with identified components and their sentences
        """
        sentences = self._preprocess_text(text)
        results = {
            "situation": [],
            "task": [],
            "action": [],
            "result": []
        }
        
        for sentence in sentences:
            # Get component matches from keywords
            components = self._analyze_sentence(sentence)
            
            # Get additional signals from verb tense
            tense_scores = self._analyze_verb_tense(sentence)
            
            # Combine evidence
            for component, confidence in components:
                additional_confidence = tense_scores.get(component, 0)
                final_confidence = min(confidence + additional_confidence, 1.0)
                
                if final_confidence > 0.3:  # Threshold for accepting a component
                    results[component].append({
                        "text": sentence,
                        "confidence": final_confidence
                    })
        
        return results

    def get_missing_components(self, analysis: Dict) -> List[str]:
        """Identify missing STAR components"""
        missing = []
        for component in ["situation", "task", "action", "result"]:
            if not analysis[component]:
                missing.append(component)
        return missing

    def generate_feedback(self, analysis: Dict) -> Dict[str, str]:
        """Generate feedback based on the analysis"""
        feedback = {}
        missing = self.get_missing_components(analysis)
        
        # Feedback for missing components
        if missing:
            feedback["missing_components"] = f"Your response is missing the following components: {', '.join(missing)}"
        
        # Feedback for each component
        for component, entries in analysis.items():
            if entries:
                avg_confidence = sum(e["confidence"] for e in entries) / len(entries)
                if avg_confidence < 0.5:
                    feedback[component] = f"The {component} could be stronger. Try to be more specific."
                elif avg_confidence < 0.7:
                    feedback[component] = f"Good {component}, but could use more detail."
                else:
                    feedback[component] = f"Strong {component} description!"
            else:
                feedback[component] = f"Try to include a clear {component} in your response."
                
        return feedback