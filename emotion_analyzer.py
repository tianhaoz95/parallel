import numpy as np
import librosa
from logger_utils import logger

class EmotionAnalyzer:
    def __init__(self):
        pass

    def analyze_audio(self, audio_path):
        """Analyzes audio for pitch, energy, and tempo features."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # 1. Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms)
            max_energy = np.max(rms)
            
            # 2. Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            # Extract dominant pitches
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0: pitch_values.append(pitch)
            
            avg_pitch = np.mean(pitch_values) if pitch_values else 0
            
            # 3. Tempo / Speech Rate
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            logger.info(f"Audio Analysis: Energy={avg_energy:.4f}, Pitch={avg_pitch:.2f}Hz, Tempo={tempo:.2f}")
            
            return {
                'energy': float(avg_energy),
                'pitch': float(avg_pitch),
                'tempo': float(tempo),
                'energy_ratio': float(max_energy / (avg_energy + 1e-6))
            }
        except Exception as e:
            logger.error(f"Failed to analyze audio emotion: {e}")
            return None

    def map_to_synthesis_params(self, analysis):
        """Maps audio features to TTS parameters (speed, temperature, etc)."""
        if not analysis:
            return {"speed": 1.0, "sway": 0.0}
            
        # Example mapping for F5-TTS or Kokoro
        # Higher energy -> slightly faster, more 'sway'
        # Higher pitch -> potentially higher emotion
        
        speed = 1.0
        if analysis['tempo'] > 140: speed = 1.1
        elif analysis['tempo'] < 80: speed = 0.9
        
        # 'sway' or 'expressiveness' proxy
        intensity = np.clip(analysis['energy'] * 10, 0, 1)
        
        return {
            "speed": speed,
            "intensity": intensity,
            "pitch_ref": analysis['pitch']
        }

if __name__ == "__main__":
    pass
