import time
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

immediate_left = tts.tts_to_file(text="Immediately turn left", file_path="immediately_turn_left.mp3", speed=2.0)
immediate_right = tts.tts_to_file(text="Immediately turn right", file_path="immediately_turn_right.mp3", speed=2.0)

possible_left = tts.tts_to_file(text="Possible left turn ahead", file_path="possible_left_turn.mp3", speed=2.0)
possible_right = tts.tts_to_file(text="Possible right turn ahead", file_path="possible_right_turn.mp3", speed=2.0)

turn_left = tts.tts_to_file(text="Turn left", file_path="turn_left.mp3", speed=2.0)
turn_right = tts.tts_to_file(text="Turn right", file_path="turn_right.mp3", speed=2.0)

continue_forward = tts.tts_to_file(text="Continue forward", file_path="continue_forward.mp3", speed=2.0)








