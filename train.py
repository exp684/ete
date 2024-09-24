from pathlib import Path
import comet_ml
import torch

from Encoder import GraphEncoder, GNNEncoder, TransformerEncoder
from Trainer import Trainer 

graph_size = 22 # Depot has two positions 0 and 1. 
n_epochs = 100
batch_size = 500 * 2
nb_train_samples = batch_size * 100 * 8
nb_val_samples = batch_size * 10
n_layers = 3
n_heads = 8
embedding_dim = 128
dim_feedforward = 512
C = 10
dropout = 0.1
learning_rate = 1e-5
seed = 1234
encoder_cls = TransformerEncoder
data_type = "Constant"  # Type of scores: Uniform, Constant, Distance.
output_dir = f"output/{encoder_cls.__name__}_{graph_size}_{data_type}"

if __name__ == "__main__":
    # Making the code device-agnostic
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('mps')
    torch.set_float32_matmul_precision('medium')
    RESUME = False  # Resume the training process from a given epoch.
    BASELINE = 'CRITIC'  # Type of baseline. EXP = Exponential, CRITIC = Critic.

    torch.set_num_threads(96)
    torch.manual_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(graph_size, n_epochs, batch_size, nb_train_samples, nb_val_samples, data_type,
                      n_layers, n_heads, embedding_dim, dim_feedforward, C,
                      dropout, learning_rate, RESUME, BASELINE, encoder_cls, output_dir)
    trainer.train()
