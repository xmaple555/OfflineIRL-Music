import editdistance
import pretty_midi
import numpy as np


def get_distance_mat(result):
    """
    Calculates the distance matrix and average score.

    Parameters:
    - result (dict): Dictionary of distance results.

    Returns:
    - D (numpy.ndarray): Distance matrix.
    - avg_score (float): Average score.
    """
    key = list(result.keys())
    x, y = len(result[key[0]]), len(key)
    D = np.zeros((y, x))
    tmp = []

    for yidx in reversed(range(y)):
        kidx = y - yidx - 1
        num_empty = x - len(result[key[kidx]])
        xidx_s = int(np.floor(num_empty / 2))
        xidx_e = int(xidx_s + len(result[key[kidx]]))

        D[yidx, xidx_s:xidx_e] = result[key[kidx]]

        # Weight function
        start_w = 0.5
        w = start_w + (1 - start_w) / y * (kidx + 1)
        distance_score = np.average(result[key[kidx]])
        weighted_D = distance_score * w
        tmp.append(weighted_D)

    return D, np.average(tmp)


def extract_pitch_interval_and_ioi(midi_file_path):
    """
    Extracts pitch intervals and IOIs (Inter-Onset Intervals) from a MIDI file.

    Parameters:
    - midi_file_path (str): Path to the MIDI file.

    Returns:
    - pitch_intervals (list): List of pitch intervals.
    - rhythm_intervals (list): List of rhythm (IOI) intervals.
    """
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Extract pitch intervals and IOIs
    pitch_intervals = []
    rhythm_intervals = []

    prev_note = None
    prev_time = None

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            current_note = note.pitch

            if prev_note is not None:
                pitch_interval = current_note - prev_note
                pitch_intervals.append(pitch_interval)

            prev_note = current_note

            current_time = note.end

            if prev_time is not None:
                ioi_interval = current_time - prev_time
                rhythm_intervals.append(ioi_interval)

            prev_time = current_time

    return pitch_intervals, rhythm_intervals


def segment_midi(midi, seg_size):
    """
    Segments a MIDI sequence into overlapping segments of a specified size.

    Parameters:
    - midi (list): List representing the MIDI sequence.
    - seg_size (int): Size of the segments.

    Returns:
    - seg_song (list): List of segmented MIDI sequences.
    """
    final_start_index = len(midi) - seg_size + 1
    seg_song = [midi[idx:idx + seg_size] for idx in range(final_start_index)]
    return seg_song


def get_min_distance(main_midi, compare_midi):
    """
    Calculates the minimum edit distance between two sets of MIDI features.

    Parameters:
    - main_midi (dict): Dictionary of main MIDI features.
    - compare_midi (dict): Dictionary of MIDI features to compare.

    Returns:
    - final_dist1 (list): List of minimum distances from main to compare.
    - final_dist2 (list): List of minimum distances from compare to main.
    """
    final_dist1, final_dist2 = [], []

    # Processing step 1: Calculate distances from main to compare
    for mfeat in main_midi:
        dist1 = [
            editdistance.eval(mfeat, cfeat) / (len(mfeat) * 2)
            for cfeat in compare_midi
        ]
        final_dist1.append(np.array(dist1).min())

    # Processing step 2: Calculate distances from compare to main
    for mfeat in compare_midi:
        dist2 = [
            editdistance.eval(mfeat, cfeat) / (len(mfeat) * 2)
            for cfeat in main_midi
        ]
        final_dist2.append(np.array(dist2).min())

    assert len(final_dist1) == len(main_midi), 'ERROR'
    assert len(final_dist2) == len(compare_midi), 'ERROR'

    return final_dist1, final_dist2


def get_paired_results(main_midi, compare_midi):
    """
    Calculates pairwise distances between two sets of MIDI features.

    Parameters:
    - main_midi (dict): Dictionary of main MIDI features.
    - compare_midi (dict): Dictionary of MIDI features to compare.

    Returns:
    - dist_mat1 (numpy.ndarray): Pairwise distance matrix for main to compare.
    - score1 (float): Overall score for main to compare.
    - dist_mat2 (numpy.ndarray): Pairwise distance matrix for compare to main.
    - score2 (float): Overall score for compare to main.
    """
    res1, res2 = {}, {}

    if len(main_midi) < len(compare_midi):
        main_midi, compare_midi = compare_midi, main_midi

    for segsize in compare_midi.keys():
        res1[str(segsize)], res2[str(segsize)] = get_min_distance(
            main_midi[str(segsize)], compare_midi[str(segsize)])

    dist_mat1, score1x = get_distance_mat(res1)
    dist_mat2, score2 = get_distance_mat(res2)
    score1, score2 = 1 - score1x, 1 - score2
    return dist_mat1, score1, dist_mat2, score2


def get_melody_distance(midi_file_path1, midi_file_path2):
    pitch_intervals1, rhythm_intervals1 = extract_pitch_interval_and_ioi(
        midi_file_path1)
    pitch_intervals2, rhythm_intervals2 = extract_pitch_interval_and_ioi(
        midi_file_path2)
    midi1_pitch = {
        str(size): segment_midi(pitch_intervals1, size)
        for size in range(3,
                          len(pitch_intervals1) + 1)
    }
    midi2_pitch = {
        str(size): segment_midi(pitch_intervals2, size)
        for size in range(3,
                          len(pitch_intervals2) + 1)
    }

    res1_pitch, s1_pitch, res2_pitch, s2_pitch = get_paired_results(
        midi1_pitch, midi2_pitch)
    return s1_pitch, s2_pitch
