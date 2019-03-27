import math
import torch
import heapq


class Sequence(object):
    def __init__(self, wids, lprob, score, att=None):
        self.wids = wids
        self.lprob = lprob
        self.score = score
        self.att = att

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score


class TopK(object):
    def __init__(self, k):
        self.K = k
        self.seqs = []

    def size(self):
        return len(self.seqs)

    def push(self, seq):
        if self.size() < self.K:
            heapq.heappush(self.seqs, seq)
        else:
            heapq.heappushpop(self.seqs, seq)

    def extract(self, descend=False):
        # this operation would empty the TopK heap
        assert self.size() != 0
        data = self.seqs
        self.seqs = []
        if descend:
            data.sort(reverse=True)
        return data


class SeqGenerator(object):
    def __init__(self, decode_step, model, src_mask, memory, eos_id, beam_size, max_seq_len=50, att=False,
                 len_norm_factor=0.0, len_norm_const=5., device_ids=None):
        self.decode_step = decode_step
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.att = att
        self.len_norm_factor = len_norm_factor
        self.len_norm_const = len_norm_const
        self.device_ids = device_ids
        self.model = model
        self.src_mask = src_mask
        self.memory = memory

    def beam_search(self, init_input):
        batch_size = init_input.size()[0]
        partial_seqs = [TopK(self.beam_size) for _ in range(batch_size)]
        complete_seqs = [TopK(self.beam_size) for _ in range(batch_size)]

        out_wids, lprobs = self.decode_step(self.model, self.src_mask, self.memory, init_input,
                                            beam_size=self.beam_size)

        # first state
        for i in range(batch_size):

            for k in range(self.beam_size):
                #                 print init_input[i].tolist()
                #                 print out_wids[i][k].data.tolist()
                #                 print lprobs[i][k]
                seq = Sequence(wids=init_input[i].tolist() + [out_wids[i][k]], lprob=lprobs[i][k], score=lprobs[i][k])

                partial_seqs[i].push(seq)

        # run beam search
        for _ in range(self.max_seq_len - 1):
            partial_seqs_list = [p.extract() for p in partial_seqs]

            # flatten all the hypothesis
            partial_seqs_flatten = [s for p in partial_seqs_list for s in p]
            next_input = torch.LongTensor([s.wids for s in partial_seqs_flatten])

            if len(next_input) == 0:
                break

            out_wids, lprobs = self.decode_step(self.model, self.src_mask, self.memory, next_input,
                                                beam_size=self.beam_size)

            idx = 0
            for i in range(batch_size):
                for seq in partial_seqs_list[i]:
                    k = 0
                    num_hyp = 0
                    #                     print idx,len(out_wids)
                    while (num_hyp < self.beam_size and idx < len(out_wids)):
                        #                         print idx,k,out_wids
                        w = out_wids[idx][k]
                        wids = seq.wids + [w]
                        lprob = seq.lprob + lprobs[idx][k]
                        score = lprob
                        k += 1
                        num_hyp += 1

                        seq_out = Sequence(wids, lprob, score)

                        if w == self.eos_id:
                            complete_seqs[i].push(seq_out)
                            num_hyp -= 1
                        else:
                            partial_seqs[i].push(seq_out)

                    idx += 1
        for i in range(batch_size):
            if complete_seqs[i].size() == 0:
                complete_seqs[i] = partial_seqs[i]

        # retrieve the top seqs
        seqs = [complete_seqs[i].extract(descend=True) for i in range(batch_size)]
        for s in seqs:
            for seq in s:
                print(seq.wids)
        return seqs


def beam_search_decode_step(model, src_mask, memory, this_input, beam_size):
    #     print this_input[0].size()
    #     this_input = torch.LongTensor(this_input)
    out_wids = []
    lprobs = []

    for i in range(this_input.size(0)):
        ys = torch.unsqueeze(this_input[i], 0)
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
        prob = model.generator(out[:, -1])
        lprob, out_wid = torch.topk(prob, min(beam_size * 2, prob.size(1)), dim=1)
        #         print lprob.data[0]
        lprobs.append(lprob.item())
        out_wids.append(out_wid.item())
        # next_word = next_word.data[0]
        # ys = torch.cat([ys, 
        #                 torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return out_wids, lprobs


def beam_search_decode(decode_step, model, src, src_mask, max_len, sos_id, eos_id, beam_size):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(sos_id).type_as(src.data)
    Decoder = SeqGenerator(decode_step, model, src_mask, memory, eos_id, beam_size, max_seq_len=max_len)
    seqs = Decoder.beam_search(ys)

    return seqs


"""
Usage:

model.eval()
src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
src_mask = torch.ones(1, 1, 10)
print(beam_search_decode(beam_search_decode_step, model, src, src_mask, max_len=10, sos_id=1, eos_id=10, beam_size=5))
"""
