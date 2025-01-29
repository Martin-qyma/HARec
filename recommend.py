import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pickle
import numpy as np
import pandas as pd
import torch.optim as optim
from utils.config import args
from models.align import HERec
from torch.optim import Optimizer
from manifolds import Hyperboloid
from rgd.rsgd import RiemannianSGD
from utils.metrics import eval_rec
from utils.sampler import WarpSampler
from utils.data_generator import Data
from utils.helper import test_output_format

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Rec:
    def __init__(self):
        self.data = Data(args.dataset)
        self.model = HERec((self.data.num_users, self.data.num_items), args).to(device)

    def test(self):
        self.model.eval()
        with open(f"./data/{args.dataset}/user_embed.pkl","rb") as f:
            user_embed = pickle.load(f)
        with open(f"./data/{args.dataset}/item_embed.pkl","rb") as f:
            item_embed = pickle.load(f)

        embeddings = torch.cat((user_embed, item_embed), dim=0)
        pred_matrix = self.model.predict(embeddings, self.data)
        results_list = eval_rec(pred_matrix, self.data)
        recall, ndcg = results_list
        print(test_output_format(recall, ndcg))

        # save the prediction
        topk = 50
        pred_matrix[self.data.user_item_csr.nonzero()] = -np.inf
        ind = np.argpartition(pred_matrix, -topk)   
        ind = ind[:, -topk:]
        arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
        pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

        prediction = {}
        for i, items in enumerate(pred_list):
            prediction[i] = items
        with open(f"../data/{self.data.dataset}/prediction.pkl", "wb") as f:
            pickle.dump(prediction, f)
        print("Prediction saved.")

    def train(self):
        # negative sampler (iterator)
        sampler = WarpSampler((self.data.num_users, self.data.num_items), self.data.adj_train, args.batch_size, args.num_neg)
        # Riemannian optimizer
        optimizer1 = RiemannianSGD(params=self.model.embedding.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, momentum=args.momentum)
        other_parameters = [param for name, param in self.model.named_parameters() if 'embedding' not in name]
        # Euclidean optimizer
        optimizer2 = optim.Adam(other_parameters, lr=args.euc_lr, weight_decay=args.weight_decay)
        num_pairs = self.data.adj_train.count_nonzero() // 2
        num_batches = int(num_pairs / args.batch_size) + 1

        best_recall = 0
        patience = 10
        for epoch in range(1, args.epochs + 1):
            avg_loss = 0.
            avg_loss_dist = 0.
            avg_loss_align = 0.
            for batch in range(num_batches):
                triples = sampler.next_batch()
                self.model.train()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                embeddings = self.model.encode(self.data.adj_train_norm)
                train_loss, loss_dist, loss_align = self.model.compute_loss(embeddings, triples)
                train_loss.backward()
                optimizer1.step()
                optimizer2.step()
                avg_loss += train_loss / num_batches
                avg_loss_dist += loss_dist / num_batches
                avg_loss_align += loss_align / num_batches

            # evaluate at the end of each batch
            avg_loss = avg_loss.detach().cpu().numpy()
            avg_loss_dist = avg_loss_dist.detach().cpu().numpy()
            avg_loss_align = avg_loss_align.detach().cpu().numpy()
            print(f'Epoch {epoch:04d} | Loss {avg_loss:.4f} | Dist {avg_loss_dist:.4f} | Align {avg_loss_align:.4f}')

            if epoch % args.eval_freq == 0:
                self.model.eval()
                embeddings = self.model.encode(self.data.adj_train_norm)
                pred_matrix = self.model.predict(embeddings, self.data)
                results_list = eval_rec(pred_matrix, self.data)
                recall, ndcg = results_list

                # apply early stop
                if best_recall < recall[0]:
                    best = recall, ndcg
                    best_recall = recall[0]
                    patience = 10
                    self.user_hyper_emb = embeddings[: self.data.num_users]
                    self.item_hyper_emb = embeddings[self.data.num_users :]
                    with open(f"./data/{args.dataset}/user_embed.pkl","wb") as f:
                        pickle.dump(self.user_hyper_emb.detach(), f)
                    with open(f"./data/{args.dataset}/item_embed.pkl","wb") as f:
                        pickle.dump(self.item_hyper_emb.detach(), f)
                else:
                    patience -= 1

                if patience == 0:
                    break
                print("Epoch: ", epoch, " Patience: ", patience)
                print(test_output_format(recall, ndcg))
                
        sampler.close()
        print("Training finished.")
        print(test_output_format(best[0], best[1]))

if __name__ == '__main__':
    rec = Rec()
    if args.mode == 'train':
        rec.train()
    elif args.mode == 'test':
        rec.test()
    else:
        print("Invalid mode. Please choose from 'train', 'test'.")