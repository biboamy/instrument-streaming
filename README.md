# Instrument streaming

This repo contains the instrument streaming model presented in the paper:
[Yun-Ning Hung, Yi-An Chen and Yi-Hsuan Yang, "MULTITASK LEARNING FOR FRAME-LEVEL INSTRUMENT RECOGNITION"](https://arxiv.org/pdf/1811.01143.pdf)


## Demo
Related websites: 
- [Project website](https://biboamy.github.io/streaming-demo/main_site/)
- [Music Transcription Overview](https://biboamy.github.io/streaming-demo/streaming/)
- [Instrument Activity Detection](https://biboamy.github.io/instrument-demo/demo.html)

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
- Run the 'output_midi.py' with the name of the pianorolls as the first arg
```
python3 output_midi.py ocean.npy
```

## Train the model
- **run.py**: start the training
- **loadData.py**: load the data from dataset
- **lib.py**: some utilities such as loss function and dataloader
- **fit.py**: trainer
- **model.py**: model's structure
- **block.py**: model's block structure
- **structure.py**: model's parameters