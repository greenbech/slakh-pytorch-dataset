from typing import List, Tuple

from pretty_midi import PrettyMIDI

from .constants import MAX_MIDI, MIN_MIDI


def parse_midis(paths: List[str]) -> List[Tuple[int, int, int, int, int]]:
    """open midi files and list of (instrument, onset, offset, note, velocity) rows"""
    data = []
    for path in paths:
        mid = PrettyMIDI(path)

        notes_out_of_range = set()
        for instrument in mid.instruments:
            for note in instrument.notes:
                if int(note.pitch) in range(MIN_MIDI, MAX_MIDI + 1):
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
                f"{len(notes_out_of_range)} notes out of MIDI range ({MIN_MIDI},{MAX_MIDI}) for file {path}. Excluded pitches: {notes_out_of_range}"
            )
    data.sort(key=lambda x: x[1])
    return data
