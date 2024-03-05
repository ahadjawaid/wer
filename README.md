# WER 

## Creating an environment

```
mamba create -n wer python=3.10
```

```
pip3 install -r requirements.txt
```

## Example Usage

`python3 wer.py --wav <path/to/wav> --text "<text input>"`

OR

```
wav_path = "path/to/wav"
text = "I like TTS!"

wer_obj = WER()
wer, cer = wer_obj = get_wer_and_cer(wav_path, text)
```
