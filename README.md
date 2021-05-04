# Slakh PyTorch Dataset

Unofficial PyTorch dataset for [Slakh](http://www.slakh.com/).

This project is a work in progress, expect breaking changes!

## Roadmap

### Automatic music transcription (AMT) usecase with audio and labels

- [x] Specify dataset split (`original`, `splits_v2`, `redux`)
- [x] Add new splits (`redux_no_pitch_bend`, ...) (Should also be filed upstream) (implemented by `skip_pitch_bend_tracks`)
- [x] Load audio `mix.flac` (all the instruments comined)
- [x] Load individual audio mixes (need to combine audio in a streaming fashion)
- [x] Specify `train`, `validation` or `test` group
- [x] Choose sequence length
- [x] Reproducable load sequences (usefull for validation group to get consistent results)
- [ ] Add more instruments (`eletric-bass`, `piano`, `guitar`, ...)
- [x] Choose between having audio in memory or stream from disk (solved by `max_files_in_memory`)
- [x] Add to pip

### Audio source separation usecase with different audio mixes
- [ ] List to come


## Usage

1. Download the Slakh dataset (see the official [website](http://www.slakh.com/)). It's about 100GB compressed so expect using some time on this point.

2. Install the Python package with pip:
```bash
pip install slakh-dataset
```

3. Convert the audio to 16 kHz (see https://github.com/ethman/slakh-utils)

4. You can use the dataset (AMT usecase):

```python
from torch.utils.data import DataLoader
from slakh_dataset import SlakhAmtDataset


dataset = SlakhAmtDataset(
    path='path/to/slakh-16khz-folder'
    split='redux', # 'splits_v2','redux-no-pitch-bend'
    audio='mix.flac', # 'individual'
    instrument='electric-bass', # or `midi_programs`
    # midi_programs=[33, 34, 35, 36, 37],
    groups=['train'],
    skip_pitch_bend_tracks=True,
    sequence_length=327680,
    max_files_in_memory=200,
)

batch_size = 8
loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

# train model on dataset...
```

## Acknowledgement

- This code is based on the dataset in [Onset and Frames](https://github.com/jongwook/onsets-and-frames) by Jong Wook Kim which is MIT Lisenced.

- Slakh http://www.slakh.com/


