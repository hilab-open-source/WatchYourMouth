import torch
import torch.distributed as dist

def ctc_collate(batch):
    video, text, pc_lens, text_lens, user, position = zip(*batch)
    video_seq = [torch.Tensor(v) for v in video]
    video_seq_padded = torch.nn.utils.rnn.pad_sequence(video_seq, batch_first=True)
    text_seq = (sum(text, []))
    text_seq = torch.LongTensor(text_seq)
    video_len = [len(v) for v in video_seq_padded]
    text_len = text_lens
    return video_seq_padded, text_seq, video_len, text_len, user, position
