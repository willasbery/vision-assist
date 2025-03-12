import time
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

turn_left = tts.tts_to_file(text="Move left", file_path="move_left.mp3", speed=2.0)
turn_right = tts.tts_to_file(text="Move right", file_path="move_right.mp3", speed=2.0)

continue_forward = tts.tts_to_file(text="Continue forward", file_path="continue_forward.mp3", speed=2.0)








