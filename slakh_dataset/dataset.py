from collections import defaultdict
import hashlib
import json
import os
import pathlib
from abc import abstractmethod
from glob import glob
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

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
from .midi import parse_midis, instrument_to_midi_programs


class Labels(NamedTuple):
    # paths to audio files (must be equal length)
    paths: List[str]
    # a matrix that contains the onset/offset/frame labels encoded as:
    # 3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
    label: torch.ByteTensor  # [num_steps, midi_bins]
    # a matrix that contains MIDI velocity values at the frame locations
    velocity: torch.ByteTensor  # [num_steps, midi_bins]


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
        label_instruments: Optional[List[str]] = None,
        label_midi_programs: Optional[List[Iterable[int]]] = None,
        groups=None,
        min_midi=MIN_MIDI,
        max_midi=MAX_MIDI,
        sequence_length=None,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
        max_harmony=None,
    ):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.label_instruments = label_instruments
        self.label_midi_programs = label_midi_programs
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.num_files = num_files
        self.max_harmony = max_harmony
        self.reproducable_load_sequences = reproducable_load_sequences

        self.file_list = []
        for group in groups:
            for file in self.files(group):
                if num_files is not None and len(self.file_list) >= num_files:
                    break
                self.file_list.append(file)
        self.file_list.sort(key=lambda x: len(x[1]), reverse=True)
        self.labels = [None] * len(self.file_list)

        self.max_files_in_memory = len(self.file_list) if max_files_in_memory < 0 else max_files_in_memory
        if self.max_files_in_memory > 0:
            self.audios = [None] * min(len(self.file_list), self.max_files_in_memory)

    def __getitem__(self, index) -> AudioAndLabels:
        track, audio_paths, tsv_path = self.file_list[index]
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

        audio_length = torchaudio.info(audio_paths[0]).num_frames
        start_frame = None
        end_frame = None
        if self.sequence_length is not None:
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

            start_frame = begin
            end_frame  = end
        else:
            if audio is None:
                audio = load_audio(audio_paths, normalize=False).to(self.device)
            else:
                audio = audio.to(self.device)
            label = labels.label.to(self.device)
            velocity = labels.velocity.to(self.device).float()

            start_frame = 0
            end_frame = audio_length

        onset = (label == 3).float()
        offset = (label == 1).float()
        frame = (label > 1).float()
        velocity = velocity.float().div_(128.0)

        return AudioAndLabels(
            track=track,
            start_time=start_frame/SAMPLE_RATE,
            end_time=end_frame/SAMPLE_RATE,
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

    def load_labels(self, audio_paths: List[str], tsv_paths: List[str]) -> Labels:
        n_keys = self.max_midi - self.min_midi + 1
        audio_length = torchaudio.info(audio_paths[0]).num_frames
        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        multi_label = torch.zeros(n_steps, n_keys, len(tsv_paths), dtype=torch.uint8)
        multi_velocity = torch.zeros(n_steps, n_keys, len(tsv_paths), dtype=torch.uint8)

        for i, tsv_path in enumerate(tsv_paths):
            saved_data_path = tsv_path.replace(".tsv", f"-{self.min_midi}-{self.max_midi}.pt")
            if os.path.exists(saved_data_path):
                label_dict = torch.load(saved_data_path)
            else:
                midi = np.atleast_2d(np.loadtxt(tsv_path, delimiter="\t", skiprows=1))
                
                label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
                velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
                if midi.size != 0:
                    if midi.shape[1] == 5:
                        for instrument, onset, offset, note, vel in midi:
                            if not (self.min_midi <= note <= self.max_midi):
                                # print(f"Skipping note {note} out of range ({self.min_midi}, {self.max_midi})")
                                continue
                            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                            onset_right = min(n_steps, left + HOPS_IN_ONSET)
                            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                            frame_right = min(n_steps, max(frame_right, onset_right))
                            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                            f = int(note) - self.min_midi
                            label[left:onset_right, f] = 3
                            label[onset_right:frame_right, f][label[onset_right:frame_right, f] == 0] = 2
                            label[frame_right:offset_right, f][label[frame_right:offset_right, f] != 3] = 1
                            velocity[left:frame_right, f] = vel
                    else:
                        raise RuntimeError(f"Unsupported tsv shape {midi.shape}")
                label_dict = dict(path=audio_paths, label=label, velocity=velocity)
                torch.save(label_dict, saved_data_path)
            multi_label[:, :, i] = label_dict["label"]
            multi_velocity[:, :, i] = label_dict["label"]
        if (self.label_instruments and isinstance(self.label_instruments, str)) or (self.label_midi_programs and isinstance(self.label_midi_programs[0], int)):
            multi_label.squeeze_(len(multi_label.shape) - 1)
            multi_velocity.squeeze_(len(multi_velocity.shape) - 1)
        return Labels(paths=audio_paths, label=multi_label, velocity=multi_velocity)

    def get_midi_notes_stats(self) -> Dict[int, int]:
        notes_states = defaultdict(int)
        for _, _, tsv_path in tqdm(self.file_list, desc="Loading midi notes stats"):
            midi = np.atleast_2d(np.loadtxt(tsv_path, delimiter="\t", skiprows=1))

            if midi.size != 0:
                if midi.shape[1] == 5:
                    for _, _, _, note, _ in midi:
                        notes_states[int(note)] += 1

        result_dict = {}
        for key in sorted(notes_states.keys()):
            result_dict[key] = notes_states[key]
        return result_dict

class SlakhAmtDataset(PianoRollAudioDataset):
    def __init__(
        self,
        path: str,
        split: str,
        audio: str,
        label_instruments: Optional[List[str]] = None,
        label_midi_programs: Optional[List[Iterable[int]]] = None,
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
        skip_missing_tracks=False,
        max_harmony=None,
    ):
        self.split = split
        self.audio = audio
        self.skip_pitch_bend_track = skip_pitch_bend_tracks
        self.skip_missing_tracks = skip_missing_tracks
        super().__init__(
            path,
            label_instruments=label_instruments,
            groups=groups if groups is not None else ["train"],
            label_midi_programs=label_midi_programs,
            min_midi=min_midi,
            max_midi=max_midi,
            sequence_length=sequence_length,
            seed=seed,
            device=device,
            num_files=num_files,
            max_files_in_memory=max_files_in_memory,
            reproducable_load_sequences=reproducable_load_sequences,
            max_harmony=max_harmony,
        )

    @classmethod
    def available_groups(cls):
        return ["train", "validation", "test"]


    def _load_track(self, track, label_midi_programs, problem_stems, pitch_bend_info) -> Optional[Tuple[List[str], List[str]]]:
        tsv_paths = []
        all_relevant_stems = set()
        for midi_programs in label_midi_programs:
            glob_path = os.path.join(self.path, "**", track)
            track_folder_list = sorted(glob(glob_path))
            if len(track_folder_list) != 1:
                if self.skip_missing_tracks:
                    print(f"Skipping track {track}")
                    return
                else:
                    raise RuntimeError(f"Missing track {track}")
            track_folder = track_folder_list[0]

            json_path = os.path.join(track_folder, "metadata.json")
            is_json = os.path.exists(json_path)
            if is_json:
                with open(json_path, 'r') as f:
                    track_metadata = json.load(f)
            else: 
                yaml_path = os.path.join(track_folder, "metadata.yaml")
                with open(yaml_path, "r") as f:
                    track_metadata = yaml.safe_load(f)

                with open(json_path, "w") as f:
                    json.dump(track_metadata, f, indent=2)
                    f.write("\n")

            relevant_stems = []
            midi_paths = []
            for stem, value in track_metadata["stems"].items():
                if not (value["audio_rendered"] and value["midi_saved"]):
                    continue
                if (-1 in midi_programs and value["is_drum"]) or value["program_num"] in midi_programs:
                    relevant_stems.append(stem)
                    midi_paths.append(os.path.join(track_folder, "MIDI", stem + ".mid"))
            relevant_stems.sort()
            if len(relevant_stems) == 0:
                return

            if track in problem_stems:
                for stem in relevant_stems:
                    if stem in problem_stems[track]:
                        print(f"Skipping track {track} because stem {stem} has error '{problem_stems[track][stem]}'")
                        return

            if self.skip_pitch_bend_track and any(
                (pitch_bend_info[track][stem]["pitch_bend"] for stem in relevant_stems)
            ):
                return

            tsv_filename = os.path.join(track_folder, "-".join(relevant_stems) + ".tsv")
            if not os.path.exists(tsv_filename):
                midi_data = parse_midis(midi_paths)
                if self.skip_pitch_bend_track and midi_data.contain_pitch_bend:
                    return
                np.savetxt(
                    tsv_filename,
                    midi_data.data,
                    fmt="%.6f",
                    delimiter="\t",
                    header="instrument\tonset\toffset\tnote\tvelocity",
                )
            tsv_paths.append(tsv_filename)

            if self.max_harmony is not None:
                label = self.load_labels([os.path.join(track_folder, "mix.flac")], [tsv_filename])
                max_harmony = torch.max(torch.sum((label.label == 3) + (label.label == 2), axis=1))
                if max_harmony > self.max_harmony:
                    print(f"Skipping track {track} due to max harmony {max_harmony}")
                    return

            for stem in relevant_stems:
                all_relevant_stems.add(stem)

        if self.audio == "individual":
            audio_paths = [os.path.join(track_folder, "stems", stem + ".flac") for stem in all_relevant_stems]
        else:
            audio_paths = [os.path.join(track_folder, f) for f in self.audio.split(",")]

        return audio_paths, tsv_paths



    def files(self, group):
        if self.label_midi_programs is None and self.label_instruments is None:
            raise RuntimeError("Both `label_midi_programs` and `label_instruments` cannot be None")
        if self.label_instruments is not None:
            if isinstance(self.label_instruments, str):
                label_midi_programs = [instrument_to_midi_programs(self.label_instruments)]
            else:
                label_midi_programs = [instrument_to_midi_programs(inst) for inst in self.label_instruments]
        else:
            if isinstance(self.label_midi_programs[0], Iterable):
                label_midi_programs = self.label_midi_programs
            else:
                label_midi_programs = [self.label_midi_programs]
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "splits", f"{self.split}.json"), "r") as f:
            split_tracks = json.load(f)

        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "splits", f"problem_stems.json"), "r") as f:
            problem_stems = json.load(f)

        pitch_bend_info = None
        if self.skip_pitch_bend_track:
            with open(
                os.path.join(pathlib.Path(__file__).parent.absolute(), "splits", f"pitch_bend_info.json"), "r"
            ) as f:
                pitch_bend_info = json.load(f)

        results = []
        for track in tqdm(split_tracks[group], desc=f"Processing group {group}"):
            result = self._load_track(track, label_midi_programs, problem_stems, pitch_bend_info)
            if result is None:
                continue
            audio_paths, tsv_paths = result
            results.append((track, audio_paths, tsv_paths))
            if self.num_files is not None and len(results) >= self.num_files:
                break

        print(f"Kept {len(results)} tracks for group {group}")
        return results
