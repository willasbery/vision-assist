import pygame
from pathlib import Path
from typing import ClassVar, Optional, Dict
from dataclasses import dataclass

from vision_assist.PathAnalyser import Corner


@dataclass
class AudioClip:
    """Represents an audio clip with its properties."""
    path: str
    volume: float = 1.0
    is_playing: bool = False


class SoundEngine:
    """
    Handles audio playback for different obstacle detections.
    """
    _instance: ClassVar[Optional['SoundEngine']] = None
    _initialized: bool = False
    
    # Define sound file paths
    SOUND_DIR = Path("audio")  # Update this to your audio directory path
    SOUND_FILES = {
        "left_obstacle": "left_obstacle.mp3",
        "right_obstacle": "right_obstacle.mp3",
        "forward_left": "forward_instruct_left_obstacle.mp3",
        "forward_right": "forward_instruct_right_obstacle.mp3"
    }
    
    def __new__(cls):
        """Ensure only one instance of SoundEngine exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the sound engine only once."""
        if not self._initialized:
            self._initialized = True
            
            # Initialize pygame mixer
            pygame.mixer.init()
            
            # Load audio clips
            self.audio_clips: Dict[str, AudioClip] = {}
            self._load_audio_clips()
            
            # Set default volume
            self.master_volume: float = 1.0
    
    def _load_audio_clips(self) -> None:
        """Load all audio clips into memory."""
        for name, filename in self.SOUND_FILES.items():
            file_path = self.SOUND_DIR / filename
            if file_path.exists():
                self.audio_clips[name] = AudioClip(str(file_path))
            else:
                print(f"Warning: Audio file not found: {file_path}")
    
    def play_sound(self, sound_type: str) -> None:
        """
        Play a specific sound type.
        
        Args:
            sound_type: Type of sound to play ('left_obstacle', 'right_obstacle', 
                       'forward_left', 'forward_right')
        """
        if sound_type in self.audio_clips:
            clip = self.audio_clips[sound_type]
            
            # Load and play the sound
            sound = pygame.mixer.Sound(clip.path)
            sound.set_volume(clip.volume * self.master_volume)
            sound.play()
            clip.is_playing = True
    
    def play_obstacle_sound(self, corner: 'Corner') -> None:
        """
        Play appropriate sound based on corner detection.
        
        Args:
            corner: Corner object containing detection information
        """
        if corner.type == 'left':
            self.play_sound('left_obstacle')
            self.play_sound('forward_left')
        elif corner.type == 'right':
            self.play_sound('right_obstacle')
            self.play_sound('forward_right')
    
    def set_volume(self, volume: float) -> None:
        """
        Set the master volume for all sounds.
        
        Args:
            volume: Volume level between 0.0 and 1.0
        """
        self.master_volume = max(0.0, min(1.0, volume))
    
    def stop_all_sounds(self) -> None:
        """Stop all currently playing sounds."""
        pygame.mixer.stop()
        for clip in self.audio_clips.values():
            clip.is_playing = False
    
    def cleanup(self) -> None:
        """Cleanup pygame mixer."""
        pygame.mixer.quit()


# Create singleton for export
sound_engine = SoundEngine()