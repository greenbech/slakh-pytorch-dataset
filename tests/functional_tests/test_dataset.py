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
    )

    audio_and_label = dataset[0]
    assert audio_and_label.track.startswith("Track")
    assert len(audio_and_label.audio.shape) == 1
    assert audio_and_label.audio.shape[0] == sequence_length

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    for batch in tqdm(loader):
        pass

    for batch in tqdm(loader):
        pass

    for batch in tqdm(loader):
        pass


def test_amt_dataset_splits_v2_mix_streaming():
    dataset = SlakhAmtDataset(
        path="data/slakh2100_flac_16k",
        split="splits_v2",
        audio="mix",
        instrument="electric-bass",
        groups=["test"],
        sequence_length=32000,
        max_files_in_memory=0,
        skip_pitch_bend_tracks=True,
        num_files=24,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    for batch in tqdm(loader):
        pass


def main():
    print("test_amt_dataset_redux_individual_in_memory()")
    test_amt_dataset_redux_individual_in_memory()
    print()

    print("test_amt_dataset_splits_v2_mix_streaming()")
    test_amt_dataset_splits_v2_mix_streaming()
    print()


if __name__ == "__main__":
    main()
