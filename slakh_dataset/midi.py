from typing import List, NamedTuple, Tuple

from pretty_midi import PrettyMIDI
from pretty_midi.utilities import pitch_bend_to_semitones

class MidiData(NamedTuple):
    data: List[Tuple[int, int, int, int, int]]
    contain_pitch_bend: bool


def instrument_to_midi_programs(instrument: str) -> List[int]:
    avaliable_instruments = ["electric-bass", "bass", "all"]
    if instrument == "electric-bass":
        return range(33, 38)
    if instrument == "bass":
        return range(32, 40)
    if instrument == "all":
        return range(0, 112)
    raise RuntimeError(f"Unsupported instrument {instrument}. Avaliable instruments: {avaliable_instruments}")


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
