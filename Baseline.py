import torch
from Model import AttentionModel


class Baseline(object):
    def __init__(self, model: AttentionModel):
        self.model = model
        self.model.set_decode_mode("greedy")

    def eval(self):
        with torch.no_grad():
            return self.model.eval()

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def evaluate(self, inputs, use_solver=False):
        if use_solver:
            raise NotImplementedError("Solver doesnt have a stable implementation yet.")
        #_, _, rollout_sol = self.model(inputs)
        _, _, rollout_sol, _ = self.model(inputs)
        #tour_lengths = self.model.split(inputs, rollout_sol)
        tour_lengths = rollout_sol
        return tour_lengths

    def set_decode_mode(self, mode):
        self.model.set_decode_mode(mode)
