import os
import torch
from torch.utils.data import DataLoader
import math
import plot
import train
from Model import AttentionModel
from OPDataset import OPDataset
import pandas as pd

if __name__ == "__main__":
    """
        This class is used to measure the performance of the best model obtained after training and evaluation.
    """
    # Making the code device-agnostic
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')

    graph_size = 21
    batch_size = 1000
    nb_test_samples = 10000
    test_samples = "random"  
    data_type = "Constant"  # Type of scores: Uniform, Constant, Distance.
    n_layers = train.n_layers
    n_heads = train.n_heads
    embedding_dim = train.embedding_dim
    dim_feedforward = train.dim_feedforward
    decode_mode = "sample"#"greedy"
    C = train.C
    dropout = train.dropout
    seed = 1234
    torch.manual_seed(seed)
    
    print("CUDA supported version by system--- ", torch.cuda.is_available())  
    print(f"CUDA version: {torch.version.cuda}")

    file_path = "test_data.pt"
    # check if file exists, generate it if not
    if not os.path.isfile(file_path):
        print('Generating test data')
        test_dataset = OPDataset(size=graph_size, num_samples=nb_test_samples, scores=data_type)
        #torch.save(test_dataset, file_path)
    else:
        print('Loading test data')
        test_dataset = torch.load(file_path)

    print("Number of test samples : ", test_dataset.num_samples)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "heuristic"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
    
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    model = AttentionModel(embedding_dim, n_layers, n_heads, dim_feedforward, C, dropout, encoder_cls=train.encoder_cls)
    model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    model_folder = "models" # Location of the best model(epoch) after evaluation.
    # load all the files and launch for each of them the evaluation
    for file in os.listdir(model_folder):
        if file.endswith(".pt"):
            print("Testing model : ", file)
            data = torch.load(os.path.join(model_folder, file))
            try:
                model.load_state_dict(data["model"])
                results = model.test_EndToEnd(test_dataloader, decode_mode=decode_mode)
                print('Results : {} in cpu = {}'.format(results["avg_ts"], results["cpu"]))
            except Exception as e:
                print(e)
