from torch.utils.data import DataLoader
from tqdm import tqdm

from slakh_dataset import SlakhAmtDataset


def test_amt_dataset_redux_individual_in_memory():
    sequence_length = 32000
    dataset = SlakhAmtDataset(
        path="data/slakh2100_flac_16k",
        split="redux",
        audio="individual",
        instrument="electric-bass",
        groups=["test"],
        sequence_length=sequence_length,
        skip_pitch_bend_tracks=True,
        max_files_in_memory=-1,
        num_files=24,
        max_harmony=2,
    )

    audio_and_label = dataset[0]
    assert audio_and_label.track.startswith("Track")
    assert len(audio_and_label.audio.shape) == 1
    assert len(dataset) == 24
    assert audio_and_label.audio.shape[0] == sequence_length
    assert audio_and_label.end_time - audio_and_label.start_time > 0

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    for batch in tqdm(loader):
        pass

    for batch in tqdm(loader):
        pass

    for batch in tqdm(loader):
        pass


def test_amt_dataset_original_mix_streaming():
    sequence_length = 32000
    min_midi = 47
    max_midi = 91
    num_files = 10
    dataset = SlakhAmtDataset(
        path="data/slakh2100_flac_16k",
        split="original",
        audio="mix.flac",
        instrument="electric-bass",
        groups=["test"],
        min_midi=min_midi,
        max_midi=max_midi,
        sequence_length=sequence_length,
        max_files_in_memory=0,
        skip_pitch_bend_tracks=True,
    )

    audio_and_label = dataset[0]
    assert audio_and_label.track.startswith("Track")
    assert len(audio_and_label.audio.shape) == 1
    assert audio_and_label.audio.shape[0] == sequence_length
    assert audio_and_label.annotation.onset.shape[1] == max_midi - min_midi + 1
    assert audio_and_label.annotation.offset.shape[1] == max_midi - min_midi + 1
    assert audio_and_label.annotation.frame.shape[1] == max_midi - min_midi + 1
    assert audio_and_label.end_time - audio_and_label.start_time > 0
    onset_len = audio_and_label.annotation.onset.shape[0]
    offset_len = audio_and_label.annotation.offset.shape[0]
    frame_len = audio_and_label.annotation.frame.shape[0]
    assert onset_len == offset_len == frame_len

    tracks = set()
    for batch in dataset:
        tracks.add(batch.track)

    assert len(tracks) == len(dataset)
    assert "Track01931" not in tracks
    assert "Track01937" not in tracks


def main():
    print("test_amt_dataset_redux_individual_in_memory()")
    test_amt_dataset_redux_individual_in_memory()
    print()

    print("test_amt_dataset_original_mix_streaming()")
    test_amt_dataset_original_mix_streaming()
    print()


if __name__ == "__main__":
    main()
