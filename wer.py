from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Tuple
import numpy as np
import librosa
import torch
import jiwer

class WER:
    def __init__(self, device: torch.device = None, target_sr: int = 16000) -> None:
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(self.device)
        self.tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.target_sr = target_sr
        
    def get_wer_and_cer(self, wav_path: str, text: str) -> Tuple[float, float]:
        '''Outputs: (WER, CER)'''
        measures = self.calculate(wav_path, text)

        return measures["wer"], measures["cer"], 
        
    def calculate(self, wav_path: str, text: str) -> float:
        wav, sr = librosa.load(wav_path)

        resampled_audio_data = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
        transcription = self.transcribe(resampled_audio_data)
        measures = calculate_measures(text, transcription)
        
        return measures
    
    def transcribe(self, wav: np.ndarray) -> str:
        inputs = self.tokenizer(wav, sampling_rate=self.target_sr, return_tensors="pt", padding="longest")
        input_values, attention_mask = inputs.input_values.to(self.device), inputs.attention_mask.to(self.device)

        logits = self.model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription
    

def calculate_measures(groundtruth, transcription):
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)
    
    measures = jiwer.compute_measures(groundtruth, transcription)
    cer_measures = jiwer.cer(groundtruth, transcription, return_dict=True)
    
    measures = {**measures, **cer_measures}

    return measures

def normalize_sentence(sentence):
    sentence = sentence.upper()
    sentence = jiwer.RemovePunctuation()(sentence)
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    sentence = jiwer.Strip()(sentence)
    sentence = sentence.upper()

    return sentence

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate WER/CER")
    parser.add_argument("--wav", type=str, help="Path to wav file")
    parser.add_argument("--text", type=str, help="Groundtruth text")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    wav_path  = args.wav
    text = args.text

    wer_obj = WER()
    wer, cer = wer_obj.get_wer_and_cer(wav_path, text)
    
    print(f"WER: {wer:.2f}, CER: {cer:.2f}")