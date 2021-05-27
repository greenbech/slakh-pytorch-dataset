from typing import List, NamedTuple, Tuple

from pretty_midi import PrettyMIDI
from pretty_midi.utilities import pitch_bend_to_semitones

class MidiData(NamedTuple):
    data: List[Tuple[int, int, int, int, int]]
    contain_pitch_bend: bool


def instrument_to_midi_programs(instrument: str) -> List[int]:
    instrument_to_midi_dict = {
        "drum": [128],
        "piano": range(8),
        "chromatic-percussion": range(8, 16),
        "organ": range(16, 24),
        "guitar": range(24, 32),
        "bass": range(32, 40),
        "strings": range(40, 47),
        "ensemble": range(48, 56),
        "brass": range(56, 64),
        "reed": range(64, 72),
        "pipe": range(72, 80),
        "synth-lead": range(80, 88),
        "synth-pad": range(88, 96),
        "synth-effects": range(96, 104),
        "ethnic": range(104, 112),
        "percussive": range(112, 120),
        "sound-effects": range(120, 128),
        "electric-bass": range(33, 38),
        "all-pitched": range(96),
    }

    if instrument not in instrument_to_midi_dict:
        raise RuntimeError(f"Unsupported instrument {instrument}. Avaliable instruments: {list(instrument_to_midi_dict.keys())}")
    else:
        return instrument_to_midi_dict[instrument]

def instrument_to_canonical_midi_program(instrument: str) -> List[int]:
    instrument_to_midi_dict = {
        "drum": 128,
        "piano": 0,
        "chromatic-percussion": 8,
        "organ": 16,
        "guitar": 26,
        "bass": 33,
        "strings": 42,
        "ensemble": 48,
        "brass": 61,
        "reed": 68,
        "pipe": 73,
        "synth-lead": 80,
        "synth-pad": 88,
        "synth-effects": 96,
        "ethnic": 104,
        "percussive": 114,
        "sound-effects": 120,
        "electric-bass": 33,
        "all-pitched": 48,
    }

    if instrument not in instrument_to_midi_dict:
        raise RuntimeError(f"Unsupported instrument {instrument}. Avaliable instruments: {list(instrument_to_midi_dict.keys())}")
    else:
        return instrument_to_midi_dict[instrument]


def parse_midis(paths: List[str]) -> MidiData:
    """open midi files and list of (instrument, onset, offset, note, velocity) rows"""
    data = []
    contain_pitch_bend = False
    bass_program_numbers = instrument_to_midi_programs('bass')
    for path in paths:
        mid = PrettyMIDI(path)

        for instrument in mid.instruments:
            if any((abs(pitch_bend_to_semitones(p.pitch)) >= 0.5 for p in instrument.pitch_bends)):
                contain_pitch_bend = True
            for note in instrument.notes:
                if instrument.program in bass_program_numbers:
                    # https://github.com/ethman/slakh-generation/issues/2
                    if note.pitch > 67:
                        continue
                    note.pitch -= 12

                if instrument.is_drum:
                    data.append(
                        (
                            128,
                            note.start,
                            note.start + 0.001,
                            int(note.pitch),
                            int(note.velocity),
                        )
                    )
                else:
                    data.append(
                        (
                            instrument.program,
                            note.start,
                            note.end,
                            int(note.pitch),
                            int(note.velocity),
                        )
                    )

    data.sort(key=lambda x: x[1])
    return MidiData(data=data, contain_pitch_bend=contain_pitch_bend)
