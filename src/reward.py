from utils import *
from cross_scape import get_melody_distance
from collections import defaultdict

max_melody_seq_len = 1024 + 1


def melody_reward(myvocab, remi_seq, melody_model, remove_ph_bc):
    reward = 0
    count = 0

    remi_seq = remi_seq.copy()

    section_start_idics = []
    section_end_idics = []
    for i in range(len(remi_seq)):
        if myvocab.id2token[remi_seq[i]] == "Section_Start":
            section_start_idics.append(i)
        elif myvocab.id2token[remi_seq[i]] == "Section_End":
            section_end_idics.append(i)

    assert len(section_start_idics) == len(section_end_idics)
    if len(section_start_idics) == 0:
        section_start_idics = [0]
        section_end_idics = [len(remi_seq) - 1]

    sections = []
    for i in range(len(section_start_idics)):
        sections.append(remi_seq[section_start_idics[i]:section_end_idics[i] +
                                 1])

        for section in sections:
            melody_section = myvocab.melody_to_interval(section)

            if remove_ph_bc:
                melody_section = myvocab.remove_ph_bc(melody_section)

            tgts = []
            for x in range(0, len(melody_section), max_melody_seq_len):
                tgts.append(melody_section[x:x + max_melody_seq_len])

            if len(tgts) >= 2:
                tgts[-1] = melody_section[-max_melody_seq_len:]

            for tgt in tgts:
                tgt_input = torch.tensor(tgt[:-1]).unsqueeze(0).cuda()
                tgt_output = torch.tensor(tgt[1:]).unsqueeze(0).cuda()

                logits = melody_model(tgt=tgt_input)

                prob = torch.softmax(logits, dim=-1)
                prob = torch.gather(prob,
                                    dim=-1,
                                    index=tgt_output.unsqueeze(-1)).squeeze()

                for i in range(len(tgt[1:])):
                    if 'Interval' in myvocab.id2token[(tgt[1:][i])]:
                        reward += prob[i].item()
                        count += 1

    if count == 0:
        return 0
    reward = reward / count
    return reward


def chord_progression_reward(myvocab, chord_model, remi_seq, remove_ph_bc):
    reward = 0
    remi_seq = remi_seq.copy()
    remi_seq = myvocab.insert_chord_label(remi_seq)
    remi_seq = myvocab.extract_chord_label(remi_seq)
    if remove_ph_bc:
        remi_seq = myvocab.remove_ph_bc(remi_seq)

    tgt_input = torch.tensor(remi_seq[:-1]).unsqueeze(0).cuda()
    tgt_output = torch.tensor(remi_seq[1:]).unsqueeze(0).cuda()

    logits = chord_model(tgt=tgt_input)

    prob = torch.softmax(logits, dim=-1)
    prob = torch.gather(prob, dim=-1, index=tgt_output.unsqueeze(-1)).squeeze()
    # reward = prob.mean().item()

    count = 0
    for i in range(len(remi_seq[1:])):
        if 'Chord' in myvocab.id2token[(remi_seq[1:][i])]:
            reward += prob[i].item()
            count += 1
    if count == 0:
        return 0

    reward /= count
    return reward


def melody_repetition(myvocab, remi_seq):
    remi_seq = remi_seq.copy()
    phrase_labels, new_phrase_indices = myvocab.extract_phrase_label(
        remi_seq, return_indices=True)

    index_dict = defaultdict(list)
    for idx, val in enumerate(phrase_labels):
        index_dict[val].append(idx)
    duplicates = {k: v for k, v in index_dict.items() if len(v) > 1}

    if len(duplicates) == 0:
        return 0

    first_dup_key = next(iter(duplicates))
    first_dup_indices = duplicates[first_dup_key]

    first_remi_seq = remi_seq[new_phrase_indices[first_dup_indices[0]]:
                              new_phrase_indices[first_dup_indices[0] + 1]]
    second_remi_seq = remi_seq[new_phrase_indices[first_dup_indices[1]]:
                               new_phrase_indices[first_dup_indices[1] + 1]]

    first_remi_seq = myvocab.remove_piano_note(first_remi_seq.copy())
    second_remi_seq = myvocab.remove_piano_note(second_remi_seq.copy())

    first_remi_seq_file = '/tmp/' + str(uuid.uuid4()) + '.mid'
    myvocab.REMIID2midi(first_remi_seq, first_remi_seq_file)
    second_remi_seq_file = '/tmp/' + str(uuid.uuid4()) + '.mid'
    myvocab.REMIID2midi(second_remi_seq, second_remi_seq_file)

    try:
        distance = get_melody_distance(first_remi_seq_file,
                                       second_remi_seq_file)
    except:
        os.remove(first_remi_seq_file)
        os.remove(second_remi_seq_file)
        return 0

    os.remove(first_remi_seq_file)
    os.remove(second_remi_seq_file)
    return distance[0]


