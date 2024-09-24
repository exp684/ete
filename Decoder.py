import math 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.distributions import Categorical
from Encoder import MultiHeadAttention
from torch import linalg as LA
from copy import deepcopy
import split_byBatch as sp

import time

class Decoder(nn.Module):
    """
        This class contains the decoder that will brings a solution for the TOP.
    """

    def __init__(self, n_head, embedding_dim, decode_mode="sample", C=10):
        super(Decoder, self).__init__()
        self.scale = math.sqrt(embedding_dim)
        self.decode_mode = decode_mode
        self.C = C  
        self.n_head = n_head

        self.vtmax = nn.Parameter(
            torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1. / embedding_dim, 1. / embedding_dim),
            requires_grad=True)
        self.vm = nn.Parameter(
            torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1. / embedding_dim, 1. / embedding_dim),
            requires_grad=True)
        
        self.W_placeholder = nn.Parameter(torch.Tensor(embedding_dim))    #Don't take the global device setting. Only keeps one variable (last).
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        
        #Projecting the context (after computing the mean of the embedding inputs)
        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        #Projecting the context (after adding the last client included into the sequence)
        self.project_context_update = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)   
        #Projecting node embeddings to compute attention (glimpse_key, glimpse_val and logit_key)
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        #Projecting node embeddings to compute attention (glimpse_key, glimpse_val and logit_key)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.glimpse = MultiHeadAttention(n_head, embedding_dim, 3 * embedding_dim, embedding_dim, embedding_dim)
        self.project_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.accelerate = Accelerator()
        (self.glimpse, self.project_k, self.cross_entropy, self.project_context, self.project_context_update, self.project_node_embeddings, self.project_out, self.vtmax, self.vm, self.W_placeholder) = \
        self.accelerate.prepare(self.glimpse, self.project_k, self.cross_entropy, self.project_context, self.project_context_update, self.project_node_embeddings, self.project_out, self.vtmax, self.vm, self.W_placeholder)

    def get_name(self):
        return f"Decoder-{self.decode_mode}"
    
    def forward(self, inputs):
        """
        :param inputs: (encoded_inputs, Coordinates(x, y), scores, Tmax, m) ([batch_size, seq_len, embedding_dim],[batch_size, seq_len, 2],[batch_size, seq_len],[batch_size],[batch_size])
        :return: raw_logits, log_probabilities, tTotalScore, solution
        """
        #encoded_inputs: decision variables.
        encoded_inputs, locations, scores, TMAX, M = inputs

        Tmax = deepcopy(TMAX) # Tour maximal length.
        m = deepcopy(M) # Tour maximal length.

        p_tmax = Tmax.unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1]
        p_m = m.unsqueeze(1).unsqueeze(1).float() # [batch_size, 1, 1] -- Valid and compulsory float conversion.

        batch_size, seq_len, embedding_dim = encoded_inputs.size()  # sel_len = nb_clients + nb_depot (2)

        h_hat = encoded_inputs.mean(-2, keepdim=True)   

        h_hat = self.project_context(h_hat) #[Batch_size, 1, embedding_dim]
        outputLog = []

        city_index = None

        mask = torch.zeros((batch_size, seq_len)).bool()

        solution = torch.tensor((batch_size, 1), dtype=torch.int64)  
        
        log_probabilities = torch.zeros(batch_size, dtype=torch.float32)

        lp_tmax = self.vtmax.repeat(batch_size, 1, 1)  # batch_size, 1, embedding_dim 
        lp_m = self.vm.repeat(batch_size, 1, 1)  # batch_size, 1, embedding_dim 

        last = torch.zeros(batch_size, 1, dtype=torch.int64)    # A sequence starts at the depot.  

        raw_logits = torch.tensor([])
        t = 0  

        # ------------------------------------------------------------------------------------
        # New variables to compute the total score per set of sequences.
        mask[:, 0] = mask[:, 1] = True   # The first two positions are the depot, thus already selected in the sequence.
        start = torch.ones(batch_size).bool()   # True, if a new route/sequence starts. 
        batch_index = torch.arange(batch_size)    # Batch indexes. Used to extract coordinates of selected clients/nodes.
        totalScore = torch.zeros(batch_size, dtype=torch.float32)   # Current route score. 
        depot_points_a = locations[:, 0:1, :] # Positions 0 and 1 are the depot. 
        distance_to_depot_a = torch.norm(locations - depot_points_a, dim=-1)
        distances = torch.norm(locations.unsqueeze(2) - locations.unsqueeze(1), dim=-1)
        # ------------------------------------------------------------------------------------
        while torch.sum(mask) < batch_size * seq_len:
            t += 1

            #------------------------------------------------------------------------------------------------------
            etmax_emb = torch.matmul(p_tmax,lp_tmax) # embedding Tmax [batch_size, 1, embedding_size]
            em_emb = torch.matmul(p_m,lp_m) # embedding m [batch_size, 1, embedding_size]
            if(t==1):                                
                mylast = self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))                  
            else:           
                exp_last = last[:, :, None].expand(batch_size, 1, encoded_inputs.size(-1))    
                mylast = encoded_inputs.gather(1, exp_last).view(batch_size, 1, -1)        
            
            embft = torch.cat((mylast, etmax_emb, em_emb), dim=1).view(batch_size, 1, -1)         
            h_hat_update = self.project_context_update(embft)   #[batch_size, 1, embedding_dim] v1, vf
            query = h_hat + h_hat_update

            glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(encoded_inputs[:, None, :, :]).chunk(3, dim=-1)
            glimpse_K = self._make_heads(glimpse_key_fixed)
            glimpse_V = self._make_heads(glimpse_val_fixed)
            logit_K = logit_key_fixed.contiguous()

            key_size = val_size = embedding_dim // self.n_head

            # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
            glimpse_Q = query.view(batch_size, 1, self.n_head, 1, key_size).permute(2, 0, 1, 3, 4)

            # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
            compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))            
            compatibility[mask[None, :, None, None, :].expand_as(compatibility)] = -math.inf

            # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
            heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

            # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)        
            glimpse = self.project_out(
                        heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, 1, self.n_head * val_size))

            # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
            logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

            # From the logits compute the probabilities by clipping, masking and softmax
            logits = torch.tanh(logits) * self.C 
            logits[mask[: ,None, :].expand_as(logits)] = -math.inf

            #------------------------------------------------------------------------------------------------------
                        
            logits = torch.log_softmax(logits, dim=-1)

            assert not torch.isnan(logits).any(), "There is a problem within the decoder...."
            
            probas = logits.exp()[:, 0, :]

            if self.decode_mode == "greedy":
                proba, city_index = self.greedy_decoding(probas)
                assert self.decode_mode == "greedy", "Greedy roll-out has not been already implemented"
            elif self.decode_mode == "sample":                
                city_index = self.new_sample_decoding(probas, mask)

            outputLog.append(logits.squeeze(1)) 
                        
            # Check insertion feasibility.
            myCityIndex = city_index.squeeze(1)            
            myLastIndex = last.squeeze(1)
            distance_last_city = distances[batch_index, myLastIndex, myCityIndex]
            distance_city_depot = distance_to_depot_a[batch_index, myCityIndex]
            distance_last_city = torch.where(start, distance_city_depot, distance_last_city)    # If the route begins, it starts at depot, otherwise it adds the distance from the last city.            
            is_valid = distance_last_city + distance_city_depot <= Tmax
            # Update Tmax.
            tempo = Tmax - distance_last_city            
            Tmax = torch.where(is_valid, tempo, Tmax)
            # Update start. If is_valid=True, then the route has started (start=False).
            both_true = is_valid & start
            start = torch.where(both_true, torch.tensor(False, dtype=torch.bool), start)

            # Update score.
            scoreSequence = scores[batch_index, myCityIndex]            
            tempoScore = totalScore + scoreSequence                        
            totalScore = torch.where(is_valid, tempoScore, totalScore)
            # Udapte m.
            distance_last_to_others = distances[batch_index, last.squeeze(1), 1:] # Shape: [batch_size, sequence]
            distances_greater_Tmax = (distance_last_to_others + distance_to_depot_a[:, 1:]) > Tmax.unsqueeze(1)            
            visited_and_greater_than_Tmax = distances_greater_Tmax | mask[:, 1:]              
            all_distances_greater_Tmax = torch.all(visited_and_greater_than_Tmax, dim=-1)     

            tempoVehicles = m - 1   # Quantity of vehicles reduced by one. 
            there_are_vehicles = tempoVehicles > 0  # if tempoVehicles=0, then no possible to reset Tmax.
            double_condition = all_distances_greater_Tmax & there_are_vehicles            
            m = torch.where(double_condition, tempoVehicles, m)
            start = start | double_condition    # Update start. Start=True, if double_condition=True or start was True. False, otherwise. 
            Tmax = torch.where(double_condition, TMAX, Tmax)    # Update Tmax.

            evalm  = m>=0
            evalTmax  = Tmax>=0
            evalTmaxUp = Tmax<=TMAX
            assert torch.all(evalm).item(), "Problem with m -- Number of vehicles is negative."
            assert torch.all(evalTmax).item(), "Problem with Tmax -- Tmax is negative."            
            assert torch.all(evalTmaxUp).item(), "Problem with Tmax -- Tmax is greater than TMAXS."

            validMask = is_valid.unsqueeze(1) == True
            last = torch.where(validMask, city_index, last)

            if (t==1):
                solution = city_index
            else:
                solution = torch.cat((solution, city_index), dim=1)
            
            # update mask.
            mask = mask.scatter(1, city_index, True)

            # update context m and Tmax.
            p_tmax = Tmax.unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1]
            p_m = m.unsqueeze(1).unsqueeze(1).float() # [batch_size, 1, 1] -- Valid and compulsory float conversion.

        #End While.
        outputLog = torch.stack(outputLog, 1)
        log_probabilities = self._calc_log_likelihood(outputLog, solution)
        
        return raw_logits, log_probabilities, totalScore, solution
    
    
    def myMasking(self, mask, locations, Tmax):
        # Depot coordinates.
        depot_points = locations[:, 1].unsqueeze(1)  # Shape: [batch_size, 1, coord_dim]
    
        # City coordinates.
        city_points = locations[:, 2:]  # Shape: [batch_size, seq_size - 2, coord_dim]
    
        distances = torch.norm(depot_points - city_points, dim=2)  # Shape: [batch_size, seq_size - 2]

        # Comparing distances.
        mask[:, 2:] = 2 * distances > Tmax.unsqueeze(1)  # Broadcasting Tmax to compare with distances

        return mask

    @staticmethod
    def greedy_decoding(probas):
        """
        :param probas: [batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        probas, city_index = torch.max(probas, dim=1)

        return probas, city_index.view(-1, 1)
    
    @staticmethod
    def new_sample_decoding(probs, mask):
        """
        :param probas: [ batch_size, seq_len]
        :param mask: [ batch_size, seq_len]
        :return: city_index: [batch_size,1]
        """
        assert (probs == probs).all(), "Probs should not contain any nans"
        city_index = probs.multinomial(1).squeeze(1)
        # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232 
        while mask.gather(1, city_index.unsqueeze(-1)).data.any():
            print('Sampled bad values, resampling!')
            city_index = probs.multinomial(1).squeeze(1)

        return city_index.view(-1, 1).to(torch.int64)

    def _make_heads(self, v):
        myTorchResult = v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_head, -1).permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)

        return (myTorchResult)
    
    def _calc_log_likelihood(self, _log_p, a):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

