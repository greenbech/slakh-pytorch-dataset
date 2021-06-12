import os
import json
from collections import defaultdict
from glob import glob
import pretty_midi
from tqdm import tqdm, trange

def main():

    slakh_path = "data/slakh2100_flac_16k"
    track_list = glob(os.path.join(slakh_path, "**", "Track*"))

    track_metadatas = {}
    for track_folder in tqdm(track_list, desc="Loading metadata"):
        json_path = os.path.join(track_folder, "metadata.json")
        with open(json_path, 'r') as f:
            track_metadatas[track_folder] = json.load(f)

    histogram_json = {}
    for midi_program in trange(0, 128, desc="MIDI Program"):
        notes_states = defaultdict(int)
        midi_paths = []
        for track_folder in track_list:
            track_metadata = track_metadatas[track_folder]
            relevant_stems = []
            for stem, value in track_metadata["stems"].items():
                if not (value["audio_rendered"] and value["midi_saved"]):
                    continue
                if value["program_num"] == midi_program:
                    relevant_stems.append(stem)
                    midi_paths.append(os.path.join(track_folder, "MIDI", stem + ".mid"))

        for midi_path in midi_paths:
            mid = pretty_midi.PrettyMIDI(midi_path)
            for instrument in mid.instruments:
                for note in instrument.notes:
                    notes_states[note.pitch] += 1

        result_dict = {}
        result_dict["instrument"] = pretty_midi.constants.INSTRUMENT_MAP[midi_program]
        result_dict["class"] = pretty_midi.constants.INSTRUMENT_CLASSES[midi_program // 8]
        result_dict["num_tracks"] = len(midi_paths)
        result_dict["num_notes"] = sum(notes_states.keys())
        result_dict["notes"] = {}
        for key in sorted(notes_states.keys()):
            result_dict["notes"][key] = notes_states[key]

        histogram_json[midi_program] = result_dict

    # drum
    notes_states = defaultdict(int)
    midi_paths = []
    for track_folder in track_list:
        track_metadata = track_metadatas[track_folder]
        relevant_stems = []
        for stem, value in track_metadata["stems"].items():
            if value["audio_rendered"] and value["midi_saved"] and value["is_drum"]:
                relevant_stems.append(stem)
                midi_paths.append(os.path.join(track_folder, "MIDI", stem + ".mid"))

    for midi_path in midi_paths:
        mid = pretty_midi.PrettyMIDI(midi_path)
        for instrument in mid.instruments:
            for note in instrument.notes:
                notes_states[note.pitch] += 1

    result_dict = {}
    result_dict["instrument"] = "Drum"
    result_dict["class"] = "Drum"
    result_dict["num_tracks"] = len(midi_paths)
    result_dict["num_notes"] = sum(notes_states.keys())
    result_dict["notes"] = {}
    for key in sorted(notes_states.keys()):
        result_dict["notes"][key] = notes_states[key]

    histogram_json[128] = result_dict

    with open('2notes_histogram.json', 'w') as f:
        json.dump(histogram_json, f, indent=2, sort_keys=False)
        f.write("\n")

main()