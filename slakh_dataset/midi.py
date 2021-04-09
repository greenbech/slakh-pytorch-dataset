from typing import List, NamedTuple, Tuple

from pretty_midi import PrettyMIDI
from pretty_midi.utilities import pitch_bend_to_semitones

from .constants import MAX_MIDI, MIN_MIDI


class MidiData(NamedTuple):
    data: List[Tuple[int, int, int, int, int]]
    contain_pitch_bend: bool


def parse_midis(paths: List[str], max_midi=MAX_MIDI, min_midi=MIN_MIDI) -> MidiData:
    """open midi files and list of (instrument, onset, offset, note, velocity) rows"""
    data = []
    contain_pitch_bend = False
    for path in paths:
        mid = PrettyMIDI(path)

        notes_out_of_range = set()
        for instrument in mid.instruments:
            if any((abs(pitch_bend_to_semitones(p.pitch)) >= 0.5 for p in instrument.pitch_bends)):
                contain_pitch_bend = True

            for note in instrument.notes:
                if int(note.pitch) in range(min_midi, max_midi + 1):
                    data.append(
                        (
                            instrument.program,
                            note.start,
                            note.end,
                            int(note.pitch),
                            int(note.velocity),
                        )
                    )
                else:
                    notes_out_of_range.add(int(note.pitch))
        if len(notes_out_of_range) > 0:
            print(
                f"{len(notes_out_of_range)} notes out of MIDI range ({min_midi},{max_midi}) for file {path}. Excluded pitches: {notes_out_of_range}"
            )

    data.sort(key=lambda x: x[1])
    return MidiData(data=data, contain_pitch_bend=contain_pitch_bend)
