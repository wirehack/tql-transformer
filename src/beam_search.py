import math
import torch
import heapq

class Sequence(object):
    def __init__(self,wids,state,lprob,score,att = None):
        self.wids = wids
        self.state = state
        self.lprob = lprob
        self.score = score
        self.att = att
    def __lt__(self,other):
        return self.score<other.score
    def __eq__(self,other):
        return self.score==other.score

class TopK(object):
    def __init__(self,k):
        self.K = k
        self.seqs = []

    def size(self):
        return len(self.seqs)

    def push(self,seq):
        if self.size()<self.K:
            heapq.heappush(self.seqs,seq)
        else:
            heapq.heappushpop(self.seqs,seq)

    def extract(self,descend=False):
        # this operation would empty the TopK heap
        assert self.size()!=0
        data = self.data
        self.data = []
        if descend:
            data.sort(reverse=True)
        return data

    class SeqGenerator(object):
        def __init__(self,decode_step,eos_id,beam_size,max_seq_len=50,att=False,len_norm_factor=0.0,len_norm_const=5.,device_ids=None):
            self.decode_step = decode_step
            self.eos_id = eos_id
            self.beam_size = beam_size
            self.max_seq_len = max_seq_len
            self.att = att
            self.len_norm_factor = len_norm_factor
            self.len_norm_const = len_norm_const
            self.device_ids = device_ids
        def beam_search(self,init_input,init_state=None):
            batch_size = init_input.size()[0]
            partial_seqs = [TopK(self.beam_size) for _ in range(batch_size)]
            complete_seqs = [TopK(self.beam_size) for _ in range(batch_size)]
            
            out_wids,lprobs,out_state = self.decode_step(init_input,init_state,beam_size=self.beam_size)

            # first state
            for i in range(batch_size):

                for k in range(self.beam_size):
                    seq = Sequence(wids = init_input[i]+[out_wids[i][k]],state = out_state[i],lprob = lprobs[i][k])

                    partial_seqs[i].push(seq)

            # run beam search
            for _ in range(self.max_seq_len-1):
                partial_seqs_list = [p.extract() for p in partial_seqs]
                
                # flatten all the hypothesis
                partial_seqs_flatten = [s for p in partial_seqs_list for s in p]
                next_input = [s.wids for s in partial_seqs_flatten]
                next_state = [s.state for s in partial_seqs_flatten]

                if len(next_input)==0:
                    break

                out_wids,lprobs,out_state = self.decode_step(next_input,next_state,beam_size=self.beam_size)

                idx =0
                for i in range(batch_size):
                    for seq in partial_seqs_list[i]:
                        state = out_state[idx]
                        k=0
                        num_hyp = 0
                        while num_hyp < self.beam_size:
                            w = out_wids[idx][k]
                            wids = seq.wids+[w]
                            lprob = seq.lprob+lprobs[idx][k]
                            score = lprob
                            k+=1
                            num_hyp+=1

                            seq_out = Sequence(wids,state,lprob,score)

                            if w == self.eos_id:
                                complete_seqs[i].push(seq_out)
                                num_hyp-=1
                            else:
                                partial_seqs[i].push(seq_out)

                        idx+=1
            for i in range(batch_size):
                if complete_seqs[i].size()==0:
                    complete_seqs[i] = partial_seqs[i]

            # retrieve the top seqs
            seqs =[complete_seqs[i].extract(descend=True)[0] for i in range(batch_size) ]

            return seqs

# def beam_searchs_decode(model, src, src_mask, max_len, start_symbol,beam_size):
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
#     for i in range(max_len-1):
#         out = model.decode(memory, src_mask, 
#                            Variable(ys), 
#                            Variable(subsequent_mask(ys.size(1))
#                                     .type_as(src.data)))
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim = 1)
#         next_word = next_word.data[0]
#         ys = torch.cat([ys, 
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#     return ys

# model.eval()
# src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
# src_mask = Variable(torch.ones(1, 1, 10) )
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
