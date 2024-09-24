import time

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from numba import jit
from torch.utils.data import DataLoader
from tqdm import tqdm

import split_byBatch as sp
from Decoder import Decoder
from Encoder import TransformerEncoder


class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, n_layers, n_head, dim_feedforward, C, dropout=0.1, encoder_cls=TransformerEncoder):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim  
        self.n_layers = n_layers    
        self.n_head = n_head    
        self.dim_feedforward = dim_feedforward  
        self.decode_mode = "sample"
        self.dropout = dropout
        self.C = C
        self.input_dim = 2


        self.encoder = encoder_cls(self.n_layers, self.n_head, self.embedding_dim,
                                          self.dim_feedforward, self.dropout)



        self.decoder = Decoder(self.n_head, self.embedding_dim, self.decode_mode, self.C)
        
    def get_name(self):
        return f"{self.encoder.get_name()}-{self.decoder.get_name()}"

    def forward(self, inputs):
        """
        :param inputs : (locations, demands, Tmax, m)
               (locations : [batch_size, seq_len, input_dim],
                scores : [batch_size, seq_len, 1],
                Tmax : [batch_size])
                m : [batch_size])

        :return: raw_logits : [batch_size, seq_len, seq_len],
                 log_prob : [batch_size],
                 solutions : [batch_size, seq_len]
        """

        inputs, scores, Tmax, m = inputs
        scor = scores.unsqueeze(-1)

        # we should pass the inputs to calculate the graph
        data = self.encoder(inputs, scor)

        # We need to pass along also the coordinates, to compute distances and to consider the constraints. 
        # The sequence scores are computed within the decoder.
        raw_logits, log_prob, totalScore, solution = self.decoder((data, inputs, scores, Tmax, m))  


        return raw_logits, log_prob, totalScore, solution

    def set_decode_mode(self, mode):
        self.decode_mode = mode
        self.decoder.decode_mode = mode

    def test_EndToEnd(self, data: DataLoader, decode_mode="sample", device='cuda'):        
        tour_scores = torch.tensor([])
        self.eval()
        self.set_decode_mode(decode_mode)
        cpu = time.time()
        runs = 1000

        for batch_id, batch in enumerate(tqdm(data)):
            locations, scores, Tmax, m = batch
            locations = locations.to(device)
            scores = scores.to(device)
            Tmax = Tmax.to(device)
            m=m.to(device)
            inputs = (locations, scores, Tmax.float(), m)
            
            #To samplig the best solution out of several runs.
            nb = len(scores) #Number of batches.
            #nb = torch.tensor(scores.size(0)).to(device='cuda') #Number of batches.
            maxs = torch.ones(nb)
            #maxs = torch.ones(nb).to(device='cuda')
            sampling_ts = torch.zeros(nb,1)
            #sampling_ts = torch.zeros(nb,1).to(device='cuda')
            for u in range(runs):
                # ----
                _, _, totalScore, solution = self(inputs)

                # Using numba to compute the split procedure ---
                #sc = totalScore
                #bts = torch.tensor(sc).view(-1)                

                #sampling_ts = torch.cat((sampling_ts, bts.unsqueeze(1)), dim=1)
                sampling_ts = torch.cat((sampling_ts, totalScore.unsqueeze(1)), dim=1)
                
            for u in range(nb):
                maxs[u] = torch.max(sampling_ts[u])     #max score over sampling.

            #tour_lengths = torch.cat((tour_lengths, btl), dim=0)
            tour_scores = torch.cat((tour_scores, maxs), dim=0)

        cpu = time.time() - cpu
        return {
            #"tour_lengths": tour_lengths,
            "tour_scores": tour_scores,
            #"avg_tl": tour_lengths.mean().item(),
            "avg_ts": tour_scores.mean().item(),
            "cpu": cpu
        }