import torch


def fix_bc_label(myvocab, words, input_phrase_labels):
    phrase_label_idx = 0
    bar_countdown = 1
    phrase_label = None
    for i in range(len(words)):
        if "Phrase" in myvocab.id2token[words[i]]:
            if phrase_label is not None and phrase_label != myvocab.id2token[
                    words[i]]:
                assert False, "Phrase inconsistency"
            elif phrase_label is None:
                phrase_label_idx += 1
                phrase_label = myvocab.id2token[words[i]]

        if "Bar" in myvocab.id2token[words[i]]:
            if bar_countdown == 1:
                bar_countdown = int(input_phrase_labels[phrase_label_idx][1:])
                words[i] = myvocab.token2id["Bar_" + str(bar_countdown)]
                phrase_label = None
            else:
                bar_countdown = bar_countdown - 1
                words[i] = myvocab.token2id["Bar_" + str(bar_countdown)]

    assert bar_countdown == 1 and phrase_label_idx == len( input_phrase_labels)
    return words


def inference(model,
              myvocab,
              max_len,
              input_phrase_labels,
              input_seq,
              t=1.0,
              p=1.0):
    model.eval()
    words = []

    word2event = myvocab.id2token

    initial_flag = True

    fail_cnt = 0

    bar_count = 0
    position_anchor = -1
    bar_countdown = 1
    phrase_label = None

    with torch.no_grad():
        while True:
            if fail_cnt:
                print('failed iterations:', fail_cnt)

            if fail_cnt > 256:
                print(
                    'model stuck ...\nPlease change a seed sand inference again!'
                )
                return None

            # prepare input
            if initial_flag:

                phrase_label_idx = 0
                prompt = [myvocab.token2id['Section_Start']]

                input_x = torch.tensor(prompt)
                words.extend(prompt)
                initial_flag = False

            else:
                input_x = torch.tensor(words[-max_len:])

            if len(words) < len(input_seq):
                word = input_seq[len(words)]
            else:

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

            if "Bar" in word2event[words[-1]]:
                if not 'Phrase' in word2event[word]:
                    print("Phrase not after Bar")
                    fail_cnt += 1
                    continue

            if 'Phrase' in word2event[word]:
                if phrase_label is not None and phrase_label != word2event[word]:
                    word = myvocab.token2id[phrase_label]
                elif phrase_label is None:
                    word = myvocab.token2id[
                        "Phrase_" +
                        str(input_phrase_labels[phrase_label_idx][0])]
                    phrase_label_idx += 1
                    phrase_label = word2event[word]

            if "Bar" in word2event[word]:
                if bar_countdown == 1 and phrase_label_idx == len(
                        input_phrase_labels):
                    word = myvocab.token2id['Section_End']
                elif bar_countdown == 1:
                    bar_countdown2 = int(
                        input_phrase_labels[phrase_label_idx][1:])
                    word = myvocab.token2id["Bar"]
                    bar_countdown = bar_countdown2
                    phrase_label = None
                else:
                    bar_countdown = bar_countdown - 1

            if 'Section_Start' == word2event[word]:
                fail_cnt += 1
                print("failed to Section_Start")
                continue

            if 'Section_End' == word2event[word]:
                if bar_countdown == 1 and phrase_label_idx < len(
                        input_phrase_labels):
                    bar_countdown2 = int(
                        input_phrase_labels[phrase_label_idx][1:])
                    word = myvocab.token2id["Bar"]
                    bar_countdown = bar_countdown2
                    phrase_label = None
                elif not (phrase_label_idx == len(input_phrase_labels) and
                          bar_countdown == 1):
                    fail_cnt += 1
                    print("failed to Section_End 3")
                    continue

            # add new event to record sequence
            words.append(word)

            if "Bar" in word2event[word]:
                bar_count += 1
                position_anchor = -1

            if 'Section_End' == word2event[word]:
                words = fix_bc_label(myvocab, words, input_phrase_labels)
                return words

            fail_cnt = 0
