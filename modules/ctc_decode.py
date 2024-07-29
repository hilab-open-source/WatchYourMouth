import torch
import difflib
import numpy as np
import editdistance
import torch.nn.functional as F
# from ctcdecode import CTCBeamDecoder
from torchaudio.functional import edit_distance

class Decoder:
    def __init__(self, labels):
        self.vocab_list = [' ']+labels
        
        # self.commands = ['START', 'RESUME', 'PAUSE', 'PREVIOUS', 'CONFIRM', 'ACCEPT',
        #                  'CANCEL', 'DISMISS', 'REJECT', 'SEARCH', 'CLOSE', 'THREE',
        #                  'SEVEN', 'EIGHT', 'HANG UP', 'VOLUME UP', 'VOLUME DOWN', 'TURN ON',
        #                  'TURN OFF', 'OK GOOGLE', 'HEY SIRI', 'ALEXA']
        
        self.commands = ['START', 'RESUME', 'PAUSE', 'PREVIOUS', 'CONFIRM', 'ACCEPT',
                         'CANCEL', 'DISMISS', 'REJECT', 'SEARCH', 'HANG UP', 'VOLUME UP',
                         'VOLUME DOWN', 'TURN ON', 'TURN OFF', 'OK GOOGLE', 'HEY SIRI',
                         'ALEXA', 'EMERGENCY', 'TAKE A SCREENSHOT', 'GET DIRECTIONS HOME',
                         'INCREASE BRIGHTNESS', 'SET A TIMER', 'SEND AN EMAIL', 'WATCH NETFLIX',
                         'CALL MOM', 'TEXT DAD', 'WHAT TIME IS IT', 'WHAT IS THE WEATHER', 'OPEN TWITTER']
        
        # self._decoder = CTCBeamDecoder(['_@']+labels[1:],
        #                                model_path=None,
        #                                alpha=0,
        #                                beta=1,
        #                                cutoff_top_n=37,
        #                                cutoff_prob=0.99,
        #                                beam_width=200,
        #                                num_processes=10,
        #                                blank_id=0,
        #                                log_probs_input=False)
    
    def convert_to_string(self, tokens, seq_len=None):
        if not seq_len:
            seq_len = tokens.size(0)
        out = []
        for i in range(seq_len):
            if len(out) == 0:
                if tokens[i] != 0:
                    out.append(tokens[i])
            else:
                if tokens[i] != 0 and tokens[i] != tokens[i - 1]:
                    out.append(tokens[i])
        return ''.join(self.vocab_list[i] for i in out)
    
    def idx_to_string(self, target, target_len, vocab):
        texts = []
        target_str = [vocab[i-1] for i in target]
        for l in target_len:
            text = ''.join(target_str[:l])
            target_str = target_str[l:]
            texts.append(text)
        return texts

    def decode_greedy(self, logits, seq_lens):
        decoded = []
        tlogits = logits.transpose(0, 1)
        _, tokens = torch.max(tlogits, 2)
        for i in range(tlogits.size(0)):
            output_str = self.convert_to_string(tokens[i], seq_lens[i])
            decoded.append(output_str)
        return decoded
    
    # def decode_beam(self, logits, seq_lens):
    #     decoded = []
    #     tlogits = logits.transpose(0, 1)
    #     beam_result, beam_scores, timesteps, out_seq_len = self._decoder.decode(tlogits.softmax(-1), seq_lens)
    #     for i in range(tlogits.size(0)):
    #         output_str = ''.join(map(lambda x: self.vocab_list[x], beam_result[i][0][:out_seq_len[i][0]]))
    #         decoded.append(output_str)
    #     return decoded
    
    def cer(self, decoded, gt):
        cer = [editdistance.eval(p[0].strip(), p[1].strip())/len(p[1].strip())
                          for p in zip(decoded, gt)]
        return np.mean(cer)
    
    def cer_sentence(self, decoded, gt):
        cer = [editdistance.eval(p[0], p[1])/len(p[1])
                          for p in zip(decoded, gt)]
        return np.mean(cer)

    def wer(self, decoded, gt):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(decoded, gt)]
        wer = [editdistance.eval(w[0], w[1])/len(w[1]) for w in word_pairs]
        return np.mean(wer)
    
    def idx_to_string_command(self, target, target_len, vocab):
        text = []
        for t in target:
            t_string = ''.join([vocab[i-1] for i in t])
            text.append(t_string)
        return text
    
    def match(self, decoded, commands):
        match_command = difflib.get_close_matches(decoded, commands, n=1, cutoff=0)
        return match_command[0]

    def accuracy(self, decoded, gt):
        word_pairs = [(' '.join(list(filter(lambda x: (x != ''), p[0].split(' ')))),
                      ' '.join(list(filter(lambda x: (x != ''), p[1].split(' ')))))
                      for p in zip(decoded, gt)]
        three_words = [(w[0], self.match(w[0], self.commands), w[1]) for w in word_pairs]
        matched = [self.match(w[0], self.commands) for w in word_pairs]
        truth = [w.strip() for w in gt]
        acc = np.sum([int(i==j) for i, j in zip(matched, truth)])/len(truth)
        return acc, matched
    
    def user_acc(self, user_order, matched, gt):
        user_order = np.array(user_order, dtype=int)
        res = user_order*[a == b.strip() for a, b in zip(matched, gt)]
        return res
    

    def compute_cer_level_distance(self, preds, gts):
        cer = 0
        length = 0
        for i in range(len(preds)):
            seq1, seq2 = preds[i].lower(), gts[i].lower()
            cer += edit_distance(seq1, seq2)
            length += len(seq2)
        return cer / length


    def compute_wer_level_distance(self, preds, gts):
        wer = 0
        length = 0
        for i in range(len(preds)):
            seq1, seq2 = preds[i].lower().split(), gts[i].lower().split()
            wer += edit_distance(seq1, seq2)
            length += len(seq2)
        return wer / length
