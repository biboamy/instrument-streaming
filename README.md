# Instrument streaming

This repo contains the instrument streaming model presented in the paper:
Yun-Ning Hung and Yi-Hsuan Yang, "MULTITASK LEARNING FOR FRAME-LEVEL INSTRUMENT RECOGNITION"

### Demo
The demo files are in the 'output_data' directory

## Musescore dataset
- **parse_data.py** contains the function to parse the Musescore dataset
- **dataset statistic.xlsx** contains the statistic of Musescore dataset 

## Run the prediction
1. Put MP3/WAV files in the "mp3" folder
2. Run the 'prediction.py' with the name of the song as the first arg
```
python3 prediction.py ocean.mp3
```
Instrument, pitch and pianorolls prediction result will be stored in the **output_data** folder 

## Convert pianorolls to MIDI 
3. Run the 'output_midi.py' with the name of the pianorolls as the first arg
```
python3 output_midi.py ocean.npy
```

## Reference

If you use the pianorolls to MIDI converter, please cite this paper:
* Hao-Wen Dong, Wen-Yi Hsiao, and Yi-Hsuan Yang, "Pypianoroll: Open Source Python Package for Handling Multitrack Pianoroll," in ISMIR Late-Breaking Demos Session, 2018.
