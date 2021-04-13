import hashlib
import json
import os
import pathlib
from abc import abstractmethod
from glob import glob
from typing import List, NamedTuple

import numpy as np
import torch
import torchaudio
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import (
    DEFAULT_DEVICE,
    HOP_LENGTH,
    HOPS_IN_OFFSET,
    HOPS_IN_ONSET,
    MAX_MIDI,
    MIN_MIDI,
    SAMPLE_RATE,
)
from .data_classes import AudioAndLabels, MusicAnnotation
from .midi import parse_midis


class Labels(NamedTuple):
    # paths to audio files (must be equal length)
    paths: List[str]
    # a matrix that contains the onset/offset/frame labels encoded as:
    # 3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
    label: torch.ByteTensor  # [num_steps, midi_bins]
    # a matrix that contains MIDI velocity values at the frame locations
    velocity: torch.ByteTensor  # [num_steps, midi_bins]


def instrument_to_midi_programs(instrument: str) -> List[int]:
    avaliable_instruments = ["electric-bass", "bass", "all"]
    if instrument == "electric-bass":
        return list(range(33, 37))
    if instrument == "bass":
        return list(range(32, 37))
    if instrument == "all":
        return list(range(0, 112))
    raise RuntimeError(f"Unsupported instrument {instrument}. Avaliable instruments: {avaliable_instruments}")


def load_audio(paths: List[str], frame_offset: int = 0, num_frames: int = -1, normalize: bool = False) -> torch.Tensor:
    audio = torchaudio.load(paths[0], frame_offset=frame_offset, num_frames=num_frames, normalize=normalize)[0]
    for path in paths[1:]:
        audio += torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames, normalize=normalize)[0]
    if audio.dtype == torch.float32 and len(audio.shape) == 2 and audio.shape[0] == 1:
        audio.squeeze_()
    else:
        raise RuntimeError(f"Unsupported tensor shape f{audio.shape} of type f{audio.dtype}")
    return audio


class PianoRollAudioDataset(Dataset):
    def __init__(
        self,
        path,
        instrument: str,
        groups=None,
        min_midi=MIN_MIDI,
        max_midi=MAX_MIDI,
        sequence_length=None,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
    ):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.instrument = instrument
        self.min_midi = min_midi
        self.max_midi = max_midi

        self.file_list = []
        for group in groups:
            for file in self.files(group):
                if num_files is not None and len(self.file_list) > num_files:
                    break
                self.file_list.append(file)
        self.labels = [None] * len(self.file_list)

        self.max_files_in_memory = len(self.file_list) if max_files_in_memory < 0 else max_files_in_memory
        if self.max_files_in_memory > 0:
            self.audios = [None] * self.max_files_in_memory
        self.reproducable_load_sequences = reproducable_load_sequences

    def __getitem__(self, index) -> AudioAndLabels:
        audio_paths, tsv_path = self.file_list[index]
        audio = None
        if index < self.max_files_in_memory:
            audio = self.audios[index]

            # The first time the audio needs to be loaded in memory
            if audio is None:
                audio = load_audio(audio_paths, normalize=False)
                self.audios[index] = audio

        labels: Labels = self.labels[index]
        # The first the labels needs to be loaded in memory
        if labels is None:
            labels = self.load_labels(audio_paths, tsv_path)
            self.labels[index] = labels

        if self.sequence_length is not None:
            audio_length = torchaudio.info(audio_paths[0]).num_frames
            possible_start_interval = audio_length - self.sequence_length
            if self.reproducable_load_sequences:
                step_begin = (
                    int(hashlib.sha256("".join(audio_paths).encode("utf-8")).hexdigest(), 16) % possible_start_interval
                )
            else:
                step_begin = self.random.randint(possible_start_interval)
            step_begin //= HOP_LENGTH

            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length
            num_frames = end - begin

            if audio is None:
                audio = load_audio(audio_paths, frame_offset=begin, num_frames=num_frames, normalize=False).to(
                    self.device
                )
            else:
                audio = audio[begin:end].to(self.device)
            label = labels.label[step_begin:step_end, :].to(self.device)
            velocity = labels.velocity[step_begin:step_end, :].to(self.device)
        else:
            if audio is None:
                audio = load_audio(audio_paths, normalize=False).to(self.device)
            else:
                audio = audio.to(self.device)
            label = labels.label.to(self.device)
            velocity = labels.velocity.to(self.device).float()

        onset = (label == 3).float()
        offset = (label == 1).float()
        frame = (label > 1).float()
        velocity = velocity.float().div_(128.0)

        track = audio_paths[0].split(os.sep)[-2]
        return AudioAndLabels(
            track=track,
            audio=audio,
            annotation=MusicAnnotation(onset=onset, offset=offset, frame=frame, velocity=velocity),
        )

    def __len__(self):
        return len(self.file_list)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load_labels(self, audio_paths: List[str], tsv_path: str) -> Labels:

        saved_data_path = tsv_path.replace(".tsv", ".pt")
        if os.path.exists(saved_data_path):
            label_dict = torch.load(saved_data_path)
        else:
            audio_length = torchaudio.info(audio_paths[0]).num_frames

            n_keys = self.max_midi - self.min_midi + 1
            n_steps = (audio_length - 1) // HOP_LENGTH + 1

            label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
            velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

            midi = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)

            if midi.size != 0:
                if midi.shape[1] == 4:
                    for onset, offset, note, vel in midi:
                        left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                        onset_right = min(n_steps, left + HOPS_IN_ONSET)
                        frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                        frame_right = min(n_steps, frame_right)
                        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                        f = int(note) - self.min_midi
                        label[left:onset_right, f] = 3
                        label[onset_right:frame_right, f] = 2
                        label[frame_right:offset_right, f] = 1
                        velocity[left:frame_right, f] = vel
                elif midi.shape[1] == 5:
                    for instrument, onset, offset, note, vel in midi:
                        left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                        onset_right = min(n_steps, left + HOPS_IN_ONSET)
                        frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                        frame_right = min(n_steps, frame_right)
                        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                        f = int(note) - MIN_MIDI
                        label[left:onset_right, f] = 3
                        label[onset_right:frame_right, f] = 2
                        label[frame_right:offset_right, f] = 1
                        velocity[left:frame_right, f] = vel
                else:
                    raise RuntimeError(f"Unsupported tsv shape {midi.shape}")
            label_dict = dict(path=audio_paths, label=label, velocity=velocity)
            torch.save(label_dict, saved_data_path)
        return Labels(paths=audio_paths, label=label_dict["label"], velocity=label_dict["velocity"])


class SlakhAmtDataset(PianoRollAudioDataset):
    def __init__(
        self,
        path: str,
        split: str,
        audio: str,
        instrument: str,
        groups=None,
        min_midi=MIN_MIDI,
        max_midi=MAX_MIDI,
        sequence_length=None,
        skip_pitch_bend_tracks=False,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
    ):
        self.split = split
        self.audio = audio
        self.skip_pitch_bend_track = skip_pitch_bend_tracks
        super().__init__(
            path,
            instrument,
            groups if groups is not None else ["train"],
            min_midi=min_midi,
            max_midi=max_midi,
            sequence_length=sequence_length,
            seed=seed,
            device=device,
            num_files=num_files,
            max_files_in_memory=max_files_in_memory,
            reproducable_load_sequences=reproducable_load_sequences,
        )

    @classmethod
    def available_groups(cls):
        return ["train", "validation", "test"]

    def files(self, group):
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "splits", f"{self.split}.json"), "r") as f:
            split_tracks = json.load(f)

        if self.instrument == "drums":
            raise NotImplementedError()
        else:
            midi_programs = instrument_to_midi_programs(self.instrument)

        if self.skip_pitch_bend_track:
            with open(
                os.path.join(pathlib.Path(__file__).parent.absolute(), "splits", f"pitch_bend_info.json"), "r"
            ) as f:
                pitch_bend_info = json.load(f)

        result = []
        for track in tqdm(split_tracks[group], desc=f"Processing groups {self.groups}"):
            glob_path = os.path.join(self.path, "**", track)
            track_folder_list = sorted(glob(glob_path))
            assert len(track_folder_list) == 1, (glob_path, track_folder_list)
            track_folder = track_folder_list[0]

            yaml_path = os.path.join(track_folder, "metadata.yaml")
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)

            relevant_stems = []
            midi_paths = []
            for stem, value in yaml_data["stems"].items():
                if (
                    value["program_num"] in midi_programs
                    and value["audio_rendered"]
                    and value["midi_saved"]
                    and not value["is_drum"]
                ):
                    relevant_stems.append(stem)
                    midi_paths.append(os.path.join(track_folder, "MIDI", stem + ".mid"))
            relevant_stems.sort()
            if len(relevant_stems) == 0:
                continue

            if self.audio == "individual":
                audio_paths = [os.path.join(track_folder, "stems", stem + ".flac") for stem in relevant_stems]
            else:
                audio_paths = [os.path.join(track_folder, "mix.flac")]

            if self.skip_pitch_bend_track and any(
                (pitch_bend_info[track][stem]["pitch_bend"] for stem in relevant_stems)
            ):
                continue

            tsv_filename = os.path.join(track_folder, "-".join(relevant_stems) + ".tsv")
            if not os.path.exists(tsv_filename):
                midi_data = parse_midis(midi_paths)
                if self.skip_pitch_bend_track and midi_data.contain_pitch_bend:
                    continue
                np.savetxt(
                    tsv_filename,
                    midi_data.data,
                    fmt="%.6f",
                    delimiter="\t",
                    header="instrument\tonset\toffset\tnote\tvelocity",
                )
            result.append((audio_paths, tsv_filename))

        print(f"Kept {len(result)} tracks for groups {self.groups}")
        return result
