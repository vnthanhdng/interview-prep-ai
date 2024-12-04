from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import spacy

class CoherenceAnalyzer:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences"""
        return self.sentence_model.encode(sentences)

    def _calculate_coherence_score(self, embeddings: np.ndarray) -> float:
        """Calculate coherence score based on sentence similarity"""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(similarity)
            
        return float(np.mean(similarities))

    def _analyze_transitions(self, text: str) -> List[Dict[str, Any]]:
        """Analyze transition words and phrases"""
        doc = self.nlp(text)
        transitions = []
        
        # Common transition words and phrases
        transition_patterns = [
            "however", "therefore", "consequently", "furthermore",
            "additionally", "moreover", "as a result", "thus",
            "for example", "specifically", "in particular"
        ]
        
        for sent in doc.sents:
            for pattern in transition_patterns:
                if pattern in sent.text.lower():
                    transitions.append({
                        "sentence": sent.text,
                        "transition": pattern
                    })
                    
        return transitions

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the coherence of the text and provide detailed feedback
        """
        # Process text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if not sentences:
            return {
                "coherence_score": 0.0,
                "feedback": "No valid sentences found in the response."
            }
            
        # Get embeddings and calculate coherence
        embeddings = self._get_sentence_embeddings(sentences)
        coherence_score = self._calculate_coherence_score(embeddings)
        
        # Analyze transitions
        transitions = self._analyze_transitions(text)
        
        # Generate feedback
        feedback = self._generate_feedback(coherence_score, transitions, len(sentences))
        
        return {
            "coherence_score": coherence_score,
            "transitions_found": len(transitions),
            "feedback": feedback,
            "details": {
                "num_sentences": len(sentences),
                "transitions": transitions
            }
        }

    def _generate_feedback(self, 
                         coherence_score: float, 
                         transitions: List[Dict], 
                         num_sentences: int) -> str:
        """Generate detailed feedback based on analysis"""
        feedback = []
        
        # Coherence score feedback
        if coherence_score < 0.5:
            feedback.append("Your response could be more coherent. Try to make clearer connections between your ideas.")
        elif coherence_score < 0.7:
            feedback.append("Your response shows decent coherence, but there's room for improvement in how ideas flow together.")
        else:
            feedback.append("Your response shows good coherence with clear connections between ideas.")
            
        # Transition words feedback
        transition_ratio = len(transitions) / max(1, num_sentences - 1)
        if transition_ratio < 0.3:
            feedback.append("Consider using more transition words to connect your ideas more smoothly.")
        elif transition_ratio < 0.6:
            feedback.append("You're using some good transitions, but adding a few more could improve flow.")
        else:
            feedback.append("Good use of transition words to connect your ideas.")
            
        return " ".join(feedback)