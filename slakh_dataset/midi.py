from typing import List, NamedTuple, Tuple

from pretty_midi import PrettyMIDI
from pretty_midi.utilities import pitch_bend_to_semitones

class MidiData(NamedTuple):
    data: List[Tuple[int, int, int, int, int]]
    contain_pitch_bend: bool

def midi_program_to_pitch_range(midi_program):
    pitch_ranges = {
        -1: (0, 1), # Rythm
        0: (21, 108), # Acoustic Grand Piano
        1: (21, 108),
        2: (21, 108),
        3: (21, 108),
        4: (21, 108),
        5: (21, 108),
        6: (21, 108),
        7: (21, 108),
        8: (21, 108), # Celesta
        9: (21, 108),
        10: (21, 108),
        11: (21, 108),
        12: (21, 108),
        13: (21, 108),
        14: (21, 108),
        15: (21, 108),
        16: (21, 108), # Drawbar Organ
        17: (21, 108),
        18: (21, 108),
        19: (21, 108),
        20: (21, 108),
        21: (21, 108),
        22: (21, 108),
        23: (21, 108),
        24: (38, 88), # Acoustic Guitar (nylon)
        25: (38, 88),
        26: (38, 88),
        27: (38, 88),
        28: (38, 88),
        29: (38, 88),
        30: (38, 88),
        31: (38, 88),
        32: (35, 79), # Acoustic Bass
        33: (35, 79),
        34: (35, 79),
        35: (35, 79),
        36: (35, 79),
        37: (35, 79),
        38: (35, 79),
        39: (35, 79),
        40: (21, 108), # Violin
        41: (21, 108),
        42: (21, 108),
        43: (21, 108),
        44: (21, 108),
        45: (21, 108),
        46: (21, 108),
        47: (21, 108),
        48: (21, 108), # String Ensemble 1
        49: (21, 108),
        50: (21, 108),
        51: (21, 108),
        52: (21, 108),
        53: (21, 108),
        54: (21, 108),
        55: (21, 108),
        56: (21, 108), # Trumpet
        57: (21, 108),
        58: (21, 108),
        59: (21, 108),
        60: (21, 108),
        61: (21, 108),
        62: (21, 108),
        63: (21, 108),
        64: (21, 108), # Soprano Sax
        65: (21, 108),
        66: (21, 108),
        67: (21, 108),
        68: (21, 108),
        69: (21, 108),
        70: (21, 108),
        71: (21, 108),
        72: (21, 108), # Piccolo
        73: (21, 108),
        74: (21, 108),
        75: (21, 108),
        76: (21, 108),
        77: (21, 108),
        78: (21, 108),
        79: (21, 108),
        80: (21, 108), # Lead 1 (square)
        81: (21, 108),
        82: (21, 108),
        83: (21, 108),
        84: (21, 108),
        85: (21, 108),
        86: (21, 108),
        87: (21, 108),
        88: (21, 108), # Pad 1 (new age)
        89: (21, 108),
        90: (21, 108),
        91: (21, 108),
        92: (21, 108),
        93: (21, 108),
        94: (21, 108),
        95: (21, 108),
        96: (21, 108), # FX 1 (rain)
        97: (21, 108),
        98: (21, 108),
        99: (21, 108),
        100: (21, 108),
        101: (21, 108),
        102: (21, 108),
        103: (21, 108),
        104: (21, 108), # Sitar
        105: (21, 108),
        106: (21, 108),
        107: (21, 108),
        108: (21, 108),
        109: (21, 108),
        110: (21, 108),
        111: (21, 108),
        112: (21, 108), # Tinkle Bell
        113: (21, 108),
        114: (21, 108),
        115: (21, 108),
        116: (21, 108),
        117: (21, 108),
        118: (21, 108),
        119: (21, 108),
        120: (21, 108), # Guitar Fret Noise
        121: (21, 108),
        122: (21, 108),
        123: (21, 108),
        124: (21, 108),
        125: (21, 108),
        126: (21, 108),
        127: (21, 108),
        128: (36, 54), # Drum
    }
    return pitch_ranges[midi_program]


def instrument_to_midi_programs(instrument: str) -> List[int]:
    instrument_to_midi_dict = {
        "rythm": [-1],
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
        "rythm": -1,
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


def parse_midis(paths: List[str], only_parse_rythm=False) -> MidiData:
    """open midi files and list of (instrument, onset, offset, note, velocity) rows"""
    data = []
    contain_pitch_bend = False
    bass_program_numbers = instrument_to_midi_programs('bass')

    if only_parse_rythm:
        mid = pretty_midi.PrettyMIDI(paths[0])
        beats = mid.get_beats()
        downbeats = mid.get_downbeats()
        for downbeat in downbeats:
            data.append(
                (
                    -1,
                    downbeat,
                    downbeat + 0.001,
                    0,
                    1,
                )
            )
        for beat in beats:
            data.append(
                (
                    -1,
                    beat,
                    beat + 0.001,
                    1,
                    1,
                )
            )
    else:
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
