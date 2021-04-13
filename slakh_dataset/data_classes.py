from typing import List, NamedTuple

import torch


class MusicAnnotation(NamedTuple):
    """An onset, offset, frame and velocity annotation"""

    onset: torch.FloatTensor  # [num_steps, midi_bins]
    offset: torch.FloatTensor  # [num_steps, midi_bins]
    frame: torch.FloatTensor  # [num_steps, midi_bins]
    velocity: torch.FloatTensor  # [num_steps, midi_bins]


class AudioAndLabels(NamedTuple):
    """Audio and label class with data that will be on GPU"""

    track: str
    audio: torch.FloatTensor  # [num_steps, n_mels]
    annotation: MusicAnnotation
