import torch

def fix_ph_bc_label(myvocab, words, input_phrase_labels):
    phrase_label_idx = 0
    bar_countdown = 1
    i = 0
    phrase_label = None
    while i < len(words):
        if "Bar" in myvocab.id2token[words[i]]:
            if bar_countdown == 1:
                bar_countdown = int(input_phrase_labels[phrase_label_idx][1:])
                words[i] = myvocab.token2id["Bar_" + str(bar_countdown)]
                phrase_label = myvocab.token2id["Phrase_" + input_phrase_labels[phrase_label_idx][0]]
                words.insert(i + 1, phrase_label)
                phrase_label_idx += 1
            else:
                bar_countdown = bar_countdown - 1
                words[i] = myvocab.token2id["Bar_" + str(bar_countdown)]
                words.insert(i + 1, phrase_label)
        i += 1

    assert bar_countdown == 1 and phrase_label_idx == len( input_phrase_labels) 
    return words

def inference(model, myvocab, max_len, bar_num, input_seq, input_phrase_labels, t=1.0, p=1.0):

    model.eval()
    words = []

    word2event = myvocab.id2token

    initial_flag = True

    fail_cnt = 0

    bar_count = 0
    position_anchor = -1

    with torch.no_grad():
        while True:
            if fail_cnt:
                print('failed iterations:', fail_cnt)

            if fail_cnt > 256:
                print(
                    'model stuck ...\nPlease change a seed sand inference again!'
                )
                return None

            if initial_flag:
                for x in input_seq:
                    if myvocab.id2token[x].startswith("Position"):
                        position_anchor = int(myvocab.id2token[x].split("_")[1])
                    if myvocab.id2token[x] == "Bar":
                        position_anchor = -1
                        bar_count += 1
                        
                words = input_seq.copy()
                input_x = torch.tensor(words)
                initial_flag = False

            else:
                input_x = torch.tensor(words[-max_len:])

            input_x = input_x.cuda()
            input_x = input_x.unsqueeze(0)

            logits = model(tgt=input_x,)

            logits = logits[:, -1, :]
            logits = torch.squeeze(logits)
            logits = logits.cpu().detach().numpy()

            probs = model.temperature(logits=logits, t=t)
            word = model.nucleus(probs=probs, p=p)

            # print("Generated new remi word {}".format(myvocab.id2token[word]))
            # skip padding
            if word in [0]:
                fail_cnt += 1
                continue

            # grammar checking ========================================================

            # check Note-On-[track] -> Note-Duration-[track]
            if 'Note-On' in word2event[
                    words[-1]] and 'Note-Duration' not in word2event[word]:
                fail_cnt += 1
                print(490)
                continue

            if 'Note-On' in word2event[
                    words[-1]] and 'Note-Duration' in word2event[word]:
                if not word2event[words[-1]].split("_")[0].split(
                        "-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-On,Duration Track Inconsistency")
                    fail_cnt += 1
                    continue

            if 'Note-Duration' in word2event[
                    word] and 'Note-On' not in word2event[words[-1]]:
                fail_cnt += 1
                print(490)
                continue

            if 'Note-Duration' in word2event[word] and 'Note-On' in word2event[
                    words[-1]]:
                if not word2event[words[-1]].split("_")[0].split(
                        "-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-On,Duration Track Inconsistency")
                    fail_cnt += 1
                    continue

            if word2event[word].startswith("Note"):
                if position_anchor == -1:
                    print("Position not yet set")
                    fail_cnt += 1
                    continue

            # check position number
            if word2event[word].startswith("Position"):
                pos = int(word2event[word].split("_")[1])
                if position_anchor > pos:
                    print("Position not increasing")
                    fail_cnt += 1
                    continue
                else:
                    position_anchor = pos

            if "Bar" in word2event[word]:
                if bar_count == bar_num:
                    words = fix_ph_bc_label(myvocab, words, input_phrase_labels)
                    return words

            words.append(word)

            if "Bar" in word2event[word]:
                bar_count += 1
                position_anchor = -1

            fail_cnt = 0
