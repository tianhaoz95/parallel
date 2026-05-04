import numpy as np
import librosa
from logger_utils import logger

class EmotionAnalyzer:
    def __init__(self):
        pass

    def analyze_audio(self, audio_path):
        """Analyzes audio for pitch, energy, and tempo features to determine emotional state."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            if len(y) == 0: return None
            
            # 1. Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms)
            
            # 2. Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if magnitudes[:, t].max() > 0.1]
            avg_pitch = np.mean(pitch_values) if pitch_values else 0
            
            # 3. Dynamic Range
            max_energy = np.max(rms)
            energy_var = np.std(rms)
            
            logger.debug(f"Audio Features: Energy={avg_energy:.4f}, Pitch={avg_pitch:.2f}Hz, Var={energy_var:.4f}")
            
            # 4. Classify Emotion for TTS Tagging
            emotion_tag = ""
            if avg_energy > 0.08:
                emotion_tag = "[shouting]" if avg_energy > 0.15 else "[excited]"
            elif avg_energy < 0.01:
                emotion_tag = "[whispering]"
            elif energy_var > 0.05:
                emotion_tag = "[expressive]"
            else:
                emotion_tag = "[neutral]"
                
            return {
                'energy': float(avg_energy),
                'pitch': float(avg_pitch),
                'tag': emotion_tag,
                'speed': 1.1 if avg_energy > 0.1 else (0.9 if avg_energy < 0.01 else 1.0)
            }
        except Exception as e:
            logger.error(f"Failed to analyze audio emotion: {e}")
            return None

if __name__ == "__main__":
    pass
