import comet_ml
import csv
import os
import time

import accelerate
import numpy as np
import torch
from accelerate import Accelerator
from comet_ml import Experiment
#from comet_ml.integration.pytorch import log_model
from scipy.stats import ttest_rel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import plot
from Baseline import Baseline
from Model import AttentionModel
from OPDataset import OPDataset


class Trainer:
    def __init__(self, graph_size, n_epochs, batch_size, nb_train_samples,
                 nb_val_samples, data_type, n_layers, n_heads, embedding_dim,
                 dim_feedforward, C, dropout, learning_rate, RESUME, BASELINE,
                 encoder_cls, output_dir
                 ):
        self.RESUME = RESUME
        self.BASELINE = BASELINE
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.nb_train_samples = nb_train_samples
        self.nb_val_samples = nb_val_samples
        self.encoder_cls = encoder_cls
        self.data_type = data_type
        self.output_dir = output_dir

        # -------------------------------------
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.C = C
        self.dropout = dropout
        # -------------------------------------

        self.max_grad_norm = 1.0
        self.accelerator = Accelerator(cpu=(not torch.cuda.is_available()))

        self.model = AttentionModel(embedding_dim, n_layers, n_heads,
                                    dim_feedforward,
                                    C, dropout, encoder_cls)  # embedding, encoder, decoder

        # This ia a rollout baseline
        baseline_model = AttentionModel(embedding_dim, n_layers, n_heads,
                                        dim_feedforward,
                                        C, dropout, encoder_cls)  # embedding, encoder, decoder
        self.baseline = Baseline(baseline_model)

        self.baseline.load_state_dict(self.model.state_dict())

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.baseline, self.optimizer = self.accelerator.prepare(self.baseline, self.optimizer)

        log_file_name = self.output_dir + "/{}-{}-logs.csv".format("top", graph_size)

        self.log = open(log_file_name, 'w', newline='')
        self.log_file = csv.writer(self.log, delimiter=",")

        self.hyper_parms = {
            "graph_size": graph_size,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "nb_train_samples": nb_train_samples,
            "nb_val_samples": nb_val_samples,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "embedding_dim": embedding_dim,
            "dim_feedforward": dim_feedforward,
            "C": C,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "RESUME": RESUME,
            "BASELINE": BASELINE,
            "encoder_cls": encoder_cls,
            "data_type": data_type
        }

        header = ["epoch", "avg_benefit", "avg_tl_epoch_train", "avg_tl_epoch_val"]
        self.log_file.writerow(header)
        self.log.flush()

        treatment_file_name = self.output_dir + "/{}-{}-Treatment.csv".format("TOP", graph_size)

        self.myLog = open(treatment_file_name, 'w', newline='')
        self.myLog_file = csv.writer(self.myLog, delimiter=",")
        for key, value in self.hyper_parms.items():
            self.myLog_file.writerow([key, value])
        self.myLog.flush()

    def train(self):
        validation_dataset = OPDataset(size=self.graph_size, num_samples=self.nb_val_samples, scores=self.data_type)
        print("Validation dataset created with {} samples".format(len(validation_dataset)))
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                           generator=torch.Generator(device=self.accelerator.device),
                                           pin_memory=self.accelerator.device.type != 'cuda')

        if os.environ.get('COMET_API_KEY') is not None:
            self.experiment = Experiment(
                api_key=os.environ.get('COMET_API_KEY'),
                project_name="top",
                workspace="sohaibafifi"
            )
            self.experiment.set_name(f"{self.model.get_name()}-{self.graph_size}-{self.BASELINE}-{self.data_type}")

            self.experiment.log_parameters(self.hyper_parms)
        else:
            self.experiment = None
        benefits = []
        avg_tour_score_batch = []
        avg_tour_score_epoch = []
        avg_tour_score_epoch_baseline = []
        avg_sc_epoch_val = []
        avg_sc_epoch_val_basline = []
        begin_epoch = 0

        if self.RESUME:
            data = torch.load(
                self.output_dir + '/SC_TOP21_Epoch_40.pt')  # Current training epoch dictionary. From this epoch the training will resume.
            if self.BASELINE == 'EXP':
                self.baseline = (data['baseline'])
            else:
                self.baseline.load_state_dict(data['baseline'])
            self.model.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            begin_epoch = data['epoch'] + 1

        for epoch in range(begin_epoch, self.n_epochs):

            cpu = time.time()

            all_tour_scores = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)
            all_tour_scores_baseline = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)

            # Put model in train mode!
            self.model.set_decode_mode("sample")
            self.model, self.optimizer, validation_dataloader = self.accelerator.prepare(self.model, self.optimizer,
                                                                                         validation_dataloader)

            if self.BASELINE == 'CRITIC':
                self.baseline.model.set_decode_mode("greedy")
                self.baseline.model = self.accelerator.prepare(self.baseline.model)

            self.model.train()

            # Generate new training data for each epoch
            train_dataset = OPDataset(size=self.graph_size, num_samples=self.nb_train_samples, scores=self.data_type)

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                          generator=torch.Generator(device=self.accelerator.device),
                                          pin_memory=self.accelerator.device.type != 'cuda')
            nb_batches = len(validation_dataloader)

            beta = 0.2  # Smoothing value

            for batch_id, batch in enumerate(tqdm(train_dataloader)):
                locations, scores, Tmax, m = batch

                inputs = (locations.to(self.accelerator.device), scores.to(self.accelerator.device),
                          Tmax.float().to(self.accelerator.device), m.to(self.accelerator.device))

                _, log_prob, totalScore, solution = self.model(inputs)

                with torch.no_grad():
                    # -----------------------------------------------------------------------------
                    tour_score = totalScore # The sequence of scores are computed within the decoder. 

                    if self.BASELINE == 'CRITIC':
                        baseline_tour_score = self.baseline.evaluate(inputs, False)
                        advantage = tour_score - baseline_tour_score[0:len(tour_score)]  # showme

                    if self.BASELINE == 'EXP':
                        if batch_id == 0:
                            if self.RESUME and begin_epoch == epoch:
                                baseline_tour_score = self.baseline
                            else:
                                baseline_tour_score = tour_score
                        else:
                            baseline_tour_score = baseline_tour_score * beta + (1 - beta) * (tour_score)
                        advantage = tour_score - baseline_tour_score

                benefit = advantage * (-log_prob)
                benefit = benefit.mean()
                if self.experiment is not None:
                    self.experiment.log_metric("benefit", benefit.item(), step=epoch * self.n_epochs + batch_id)
                self.optimizer.zero_grad()
                self.accelerator.backward(benefit)

                self.optimizer.step()
                self.accelerator.wait_for_everyone()

                benefits.append(benefit.item())
                avg_tour_score_batch.append(tour_score.mean().item())
                all_tour_scores = torch.cat((all_tour_scores, tour_score), dim=0)
                all_tour_scores_baseline = torch.cat((all_tour_scores_baseline, baseline_tour_score), dim=0)

            avg_tour_score_epoch.append(all_tour_scores.mean().item())
            avg_tour_score_epoch_baseline.append(all_tour_scores_baseline.mean().item())

            print(
                "\nEpoch: {}\t\nAverage tour score model : {}\nAverage tour score baseline : {}\n".format(
                    epoch + 1, all_tour_scores.mean(), all_tour_scores_baseline.mean()
                ))

            print("Validation and rollout update check\n")
            avg_benefit = np.mean(benefits[-nb_batches:])

            if self.BASELINE == 'CRITIC':  # Critic roll-out baseline.
                # t-test :
                self.model.set_decode_mode("greedy")
                self.baseline.set_decode_mode("greedy")
                self.model.eval()
                self.baseline.eval()
                with torch.no_grad():
                    rollout_tl = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)
                    policy_tl = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)
                    for batch_id, batch in enumerate(tqdm(validation_dataloader)):
                        locations, scores, Tmax, m = batch

                        inputs = (locations, scores, Tmax.float(), m)

                        _, log_prob, totalScore, solution = self.model(inputs)
                        tour_score = totalScore
                        baseline_tour_score = self.baseline.evaluate(inputs, False)

                        rollout_tl = torch.cat((rollout_tl, baseline_tour_score.view(-1)), dim=0)
                        policy_tl = torch.cat((policy_tl, tour_score.view(-1)), dim=0)

                    rollout_tl = rollout_tl.cpu().numpy()
                    policy_tl = policy_tl.cpu().numpy()

                    avg_ptl = np.mean(policy_tl)
                    avg_rtl = np.mean(rollout_tl)

                    avg_sc_epoch_val.append(avg_ptl.item())
                    avg_sc_epoch_val_basline.append(avg_rtl.item())

                    cpu = time.time() - cpu
                    print(
                        "CPU: {}\n"
                        "Benefit: {}\n"
                        "Average tour score by policy: {}\n"
                        "Average tour score by rollout: {}\n".format(cpu, avg_benefit, avg_ptl, avg_rtl))

                    self.log_file.writerow([epoch, avg_benefit,
                                            avg_tour_score_epoch[-1],
                                            avg_ptl.item()
                                            ])
                    if self.experiment is not None:
                        self.experiment.log_metric("avg_benefit", avg_benefit, epoch=epoch)
                        self.experiment.log_metric("avg_tour_score_epoch", avg_tour_score_epoch[-1], epoch=epoch)
                        self.experiment.log_metric("avg_tour_score_epoch_baseline", avg_tour_score_epoch_baseline[-1],
                                                   epoch=epoch)

                    self.log.flush()

                    if (avg_ptl - avg_rtl) > 0:
                        # t-test
                        _, pvalue = ttest_rel(rollout_tl, policy_tl)
                        pvalue = pvalue / 2  # one-sided ttest [refer to the original implementation]
                        if pvalue < 0.05:
                            print("Rollout network update...\n")
                            self.baseline.load_state_dict(self.model.state_dict())
                            print("Generate new validation dataset\n")

                            validation_dataset = OPDataset(size=self.graph_size, num_samples=self.nb_val_samples,
                                                           scores=self.data_type)

                            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size,
                                                               shuffle=False,
                                                               num_workers=0,
                                                               generator=torch.Generator(
                                                                   device=self.accelerator.device),
                                                               pin_memory=self.accelerator.device.type != 'cuda'
                                                               )
            else:  # EXP baseline.
                self.model.set_decode_mode("sample")
                self.model.eval()
                with torch.no_grad():
                    policy_tl = torch.tensor([], dtype=torch.float32)
                    for batch_id, batch in enumerate(tqdm(validation_dataloader)):
                        locations, scores, Tmax, m = batch
                        inputs = (locations, scores, Tmax.float(), m)

                        _, log_prob, totalScore, solution = self.model(inputs)

                        tour_score = totalScore
                        policy_tl = torch.cat((policy_tl, tour_score.view(-1)), dim=0)

                    policy_tl = policy_tl.cpu().numpy()
                    avg_ptl = np.mean(policy_tl)
                    avg_sc_epoch_val.append(avg_ptl.item())
                    avg_sc_epoch_val_basline.append(avg_ptl.item())

                    cpu = time.time() - cpu
                    print(
                        "CPU: {}\n"
                        "Benefit: {}\n"
                        "Average tour score by policy: {}\n".format(cpu, avg_benefit, avg_ptl))

                    self.log_file.writerow([epoch, avg_benefit,
                                            avg_tour_score_epoch[-1],
                                            avg_ptl.item()
                                            ])

                    self.log.flush()
            if epoch >= 0: #epoch % 10 == 0 or epoch == self.n_epochs - 1: 
                model_name = self.output_dir + "/SC_{}{}_Epoch_{}.pt".format("TOP", self.graph_size, epoch + 1)
                if self.BASELINE == 'CRITIC':
                    torch.save({
                        'epoch': epoch,
                        'baseline': self.baseline.state_dict(),
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }, model_name)
                else:  # Exponential baseline.
                    torch.save({
                        'epoch': epoch,
                        'baseline': baseline_tour_score,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }, model_name)

                plot.plot_stats(benefits,
                                "{}-SC-Benefits per batch {}".format("op", self.graph_size),
                                "Batch", "Benefit", self.output_dir)
                plot.plot_stats2(
                    avg_tour_score_epoch,
                    avg_tour_score_epoch_baseline,
                    "{} Average tour score per epoch train {}".format("op", self.graph_size),
                    "Epoch", "Average tour score", self.output_dir)
                plot.plot_stats(
                    avg_tour_score_epoch,
                    "{} Average tour score per epoch train (Original) {}".format("op", self.graph_size),
                    "Epoch", "Average tour score", self.output_dir)
                plot.plot_stats(
                    avg_tour_score_batch,
                    "{} Average tour score per batch train {}".format("op", self.graph_size),
                    "Batch", "Average tour score", self.output_dir)
                plot.plot_stats2(
                    avg_sc_epoch_val,
                    avg_sc_epoch_val_basline,
                    "{} Average tour score per epoch validation {}".format("op", self.graph_size),
                    "Epoch", "Average tour score", self.output_dir)
                #if self.experiment is not None:
                #    log_model(self.experiment, model=self.model, model_name="model-{}".format(epoch))

            if self.experiment is not None:
                self.experiment.log_metric("avg_benefit", avg_benefit, epoch=epoch)
                self.experiment.log_metric("avg_tour_score_epoch", avg_tour_score_epoch[-1], epoch=epoch)
                self.experiment.log_metric("avg_tour_score_epoch_baseline", avg_tour_score_epoch_baseline[-1],
                                           epoch=epoch)
                self.experiment.log_metric("avg_tour_score_val", avg_sc_epoch_val[-1], epoch=epoch)
                self.experiment.log_metric("avg_tour_score_val_baseline", avg_sc_epoch_val_basline[-1], epoch=epoch)
