import torch


def fix_ph_label(myvocab, words, input_phrase_labels):
    phrase_label_idx = 0
    bar_countdown = 1
    i = 0
    phrase_label = None
    while i < len(words):
        if "Bar" in myvocab.id2token[words[i]]:
            if bar_countdown == 1:
                bar_countdown = int(input_phrase_labels[phrase_label_idx][1:])
                assert int(myvocab.id2token[words[i]].split("_")[1]) == bar_countdown
                phrase_label = myvocab.token2id["Phrase_" + input_phrase_labels[phrase_label_idx][0]]
                words.insert(i + 1, phrase_label)
                phrase_label_idx += 1
            else:
                bar_countdown = bar_countdown - 1
                assert int(myvocab.id2token[words[i]].split("_")[1]) == bar_countdown
                words.insert(i + 1, phrase_label)
        i += 1

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


            if "Bar" in word2event[word]:
                bar_countdown2 = int(word2event[word].split("_")[1])

                if bar_countdown == 1 and phrase_label_idx == len(
                        input_phrase_labels):
                    word = myvocab.token2id['Section_End']
                elif bar_countdown == 1:
                    bar_countdown2 = int(
                        input_phrase_labels[phrase_label_idx][1:])
                    word = myvocab.token2id["Bar_" + str(bar_countdown2)]
                    bar_countdown = bar_countdown2
                    phrase_label_idx += 1
                elif bar_countdown2 != bar_countdown - 1:
                    print("Bar count inconsistency")
                    fail_cnt += 1
                    continue
                else:
                    bar_countdown = bar_countdown2

            if 'Section_Start' == word2event[word]:
                fail_cnt += 1
                print("failed to Section_Start")
                continue

            if 'Section_End' == word2event[word]:
                if bar_countdown == 1 and phrase_label_idx < len(
                        input_phrase_labels):
                    bar_countdown2 = int(
                        input_phrase_labels[phrase_label_idx][1:])
                    word = myvocab.token2id["Bar_" + str(bar_countdown2)]
                    bar_countdown = bar_countdown2
                    phrase_label_idx += 1
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
                words = fix_ph_label(myvocab, words, input_phrase_labels)
                return words

            fail_cnt = 0