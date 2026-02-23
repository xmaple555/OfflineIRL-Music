import torch
from theme_preprocess.vocab import Vocab as ThemeVocab
mythemevocab = ThemeVocab()

def inference(model, n_bars,theme_seq,prompt=None, t=1.0, p=1.0):
    """inference function

    Args:
        n_bars (int): numbers of bar to generate
        strategies (dict): inferencing strategies
        params (dict): parameters for inferencing strategies
        theme_seq (list): given theme condition
        prompt (list, optional): initial tokens fed to the theme transformer. Defaults to None.

    Returns:
        list: token sequence of generated music
    """
    model.eval()
    words = [[]]

    word2event = mythemevocab.id2token

    initial_flag, initial_cnt = True, 0
    generated_bars = 0
    
    fail_cnt = 0
    subbeats_accumulate = 0

    input_theme = torch.tensor(theme_seq)
    input_theme = input_theme.reshape((-1,1))
    input_theme = input_theme.cuda()

    label_list = []

    previous_labeled = False

    last_theme_bar_idx = -1

    new_motif_tmp_array = []
    
    bar_count = 0
    position_anchor = -1

    with torch.no_grad():
        while True:
            # print("events #{} Generating Bars #{}/{}".format(len(words[0]),bar_count ,bar_count),end='\r')
            if fail_cnt:
                print ('failed iterations:', fail_cnt)
            
            if fail_cnt >1024:
                print ('model stuck ...\nPlease change a seed sand inference again!')
                return None

            # prepare input
            if initial_flag:
                if not prompt == None:
                    # prompt given
                    input_x = torch.tensor(prompt)
                    words[0].extend(prompt)
                    label_list = [0]*len(prompt)
                    for i,x in enumerate(prompt):
                        if mythemevocab.id2token[x] == "Theme_Start":
                            previous_labeled = True
                        elif mythemevocab.id2token[x] == "Theme_End":
                            previous_labeled = False
                        if previous_labeled:
                            if i == 0:
                                label_list[i] = 1
                            else:
                                label_list[i] = label_list[i-1] +  1
                        if mythemevocab.id2token[x].startswith("Position"):
                            position_anchor = int(mythemevocab.id2token[x].split("_")[1])
                        if mythemevocab.id2token[x] == "Bar":
                            position_anchor = - 1
                            bar_count += 1
                    
                    label_input = torch.tensor(label_list)
                else:
                    # no prompt given
                    input_x = torch.tensor([theme_seq[0]])
                    label_list = [0]
                    words[0].append(theme_seq[0])
                    if mythemevocab.id2token[theme_seq[0]] == "Theme_Start":
                        previous_labeled = True
                    label_input = torch.tensor(label_list)

                initial_flag = False
            else:
                input_x = torch.tensor(words[0][-512:])
                label_input = torch.tensor(label_list[-512:])



            input_x = input_x.reshape((-1,1))
            label_input = label_input.reshape((-1,1))
            
            input_x_att_msk = model.transformer_model.generate_square_subsequent_mask(input_x.shape[0])
            input_x = input_x.cuda()
            label_input = label_input.cuda()
            input_x_att_msk = input_x_att_msk.cuda()


            logits = model(
                src=input_theme,
                tgt=input_x,
                tgt_label=label_input,
                tgt_mask = input_x_att_msk
            )   
            
            logits = logits[-1:]
            logits = torch.squeeze(logits)
            logits = logits.cpu().numpy()

            probs = model.temperature(logits=logits, t=t)
            word = model.nucleus(probs=probs, p=p)


            # print("Generated new remi word {}".format(mythemevocab.id2token[word]))
            # skip padding
            if word in [0]:
                fail_cnt += 1
                continue
            
            # grammar checking ========================================================

            #  check Theme_Start -> Bar
            if 'Theme_Start' in  word2event[words[0][-1]] and 'Bar' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue

            #  check Theme_End -> Bar
            if 'Theme_End' in  word2event[words[0][-1]] and 'Bar' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue

            # check Note-On-[track] -> Note-Duration-[track]
            if 'Note-On' in  word2event[words[0][-1]] and 'Note-Duration' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-On' in  word2event[words[0][-1]] and 'Note-Duration' in  word2event[word]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-On,Duration Track Inconsistency")
                    continue

            if 'Note-Duration' in  word2event[word] and 'Note-On' not in  word2event[words[0][-1]]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-Duration' in  word2event[word] and 'Note-On' in  word2event[words[0][-1]]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-On,Duration Track Inconsistency")
                    continue

            # check Note-Duration-[track] -> Note-Velocity-[track]
            if 'Note-Duration' in  word2event[words[0][-1]] and 'Note-Velocity' not in  word2event[word]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-Duration' in  word2event[words[0][-1]] and 'Note-Velocity' in  word2event[word]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-Duration,Velocity Track Inconsistency")
                    continue

            if 'Note-Velocity' in  word2event[word] and 'Note-Duration' not in  word2event[words[0][-1]]:
                fail_cnt += 1
                print(490)
                continue
            if 'Note-Velocity' in  word2event[word] and 'Note-Duration' in  word2event[words[0][-1]]:
                if not word2event[words[0][-1]].split("_")[0].split("-")[2] == word2event[word].split("_")[0].split("-")[2]:
                    print("Note-Duration,Velocity Track Inconsistency")
                    continue
            

            if word2event[word].startswith("Tempo") or word2event[word].startswith("Note"):
                if position_anchor == -1:
                    print("Position not yet set")
                    fail_cnt += 1 
                    continue
            
            # check position number
            if word2event[word].startswith("Position"):
                pos = int(word2event[word].split("_")[1])
                if position_anchor == pos:
                    print("Position not increasing")
                    fail_cnt += 1 
                    continue
                else:
                    position_anchor = pos

            
            # check theme region
            if mythemevocab.id2token[word].startswith("Theme"):
                if mythemevocab.id2token[word] == "Theme_Start" and not previous_labeled:
                    previous_labeled = True
                    last_theme_bar_idx = bar_count

                elif mythemevocab.id2token[word] == "Theme_End" and previous_labeled:
                    previous_labeled = False
                else:
                    print("Theme region error")
                    fail_cnt += 1
                    continue

            if word2event[word] == "Bar":
                if bar_count == n_bars:
                    return words[0]
                
            # add new event to record sequence
            words[0].append(word)
            if previous_labeled:
                label_list.append(label_list[-1]+1)
            else:
                label_list.append(0)

            if word2event[word] == "Bar":
                bar_count += 1
                position_anchor = -1

            fail_cnt = 0

