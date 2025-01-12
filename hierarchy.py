import torch
import pickle
import numpy as np
from utils.config import args
from models.balance import Balance
from rgd.rsgd import RiemannianSGD
from utils.data_generator import Data
from utils.helper import hierarchy_output_format
from torch.utils.data import Dataset, DataLoader
from models.clustering import HierarchicalClustering
from utils.metrics import EPC, Shannon_entropy, Diversity, recall_at_k, ndcg_at_k, eval_rec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Hierarchy:
    def __init__(self):
        self.data = Data(args.dataset)
        self.prediction = pickle.load(open(f"./data/{args.dataset}/prediction.pkl", "rb"))

    def build_tree(self):
        user_embeds = pickle.load(open(f"./data/{args.dataset}/user_embed.pkl","rb"))
        item_embeds = pickle.load(open(f"./data/{args.dataset}/item_embed.pkl","rb"))
        embeds = torch.cat((user_embeds, item_embeds), dim=0).to(device)
        hierarchy_tree = HierarchicalClustering(embeds, k=args.k)
        hierarchy_tree.print_tree()
        with open(f"./data/{args.dataset}/hierarchy_tree.pkl", "wb") as f:
            pickle.dump(hierarchy_tree.tree, f)
        print(f"Hierarchy tree saved.")

    def evaluate(self):
        item_emb = pickle.load(open(f"./data/{args.dataset}/item_embed.pkl", "rb"))

        recall = []
        ndcg = []
        diversity = []
        shannon = []
        epc = []
        item_degree = {}
        for uid, items in self.data.train_dict.items():
            for item in items:
                if item in item_degree:
                    item_degree[item] += 1
                else:
                    item_degree[item] = 1
        for k in [20]:
            recommended_items = []
            for uid, items in self.data.train_dict.items():
                if uid in self.prediction:
                    recommended_items.extend(self.prediction[uid][:k])

            recall.append(recall_at_k(self.data.test_dict, self.prediction, k))
            ndcg.append(ndcg_at_k(self.data.test_dict, self.prediction, k))
            diversity.append(Diversity(self.prediction, item_emb, k))
            shannon.append(Shannon_entropy(recommended_items))
            epc.append(EPC(recommended_items, item_degree))

        print("Dataset: ", args.dataset)
        print('R@20\tN@20\tDiv@20\tH@20\tEPC@20')
        print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(recall[0], ndcg[0], diversity[0], shannon[0], epc[0]))

    def balance(self):
        self.model = Balance(self.data)
        self.prediction = self.model.predict(args.temperature, args.layer)
        self.evaluate()
        print("Temperature:", args.temperature, "Layer:", args.layer, "Max layer:", self.model.max_layer)
    
    def case_study(self):
        model = Balance(self.data)
        user_embed = pickle.load(open(f"./data/{args.dataset}/user_embed.pkl", "rb"))
        item_embed = pickle.load(open(f"./data/{args.dataset}/item_embed.pkl", "rb"))
        usr_prf = pickle.load(open(f"./data/{args.dataset}/usr_prf.pkl", "rb"))
        itm_prf = pickle.load(open(f"./data/{args.dataset}/itm_prf.pkl", "rb"))
        
        user = 100
        print(f"User Index: {user}")
        print(f"User Profile:\n{usr_prf[user]['profile']}")
        item = model.find_index_with_lca(item_embed[user], 5)
        print(f"Item Index: {item}")
        print(f"Item Profile:\n{itm_prf[item]['profile']}")

if __name__ == "__main__":
    hierarchy = Hierarchy()
    if args.mode == 'build':
        hierarchy.build_tree()
    elif args.mode == 'evaluate':
        hierarchy.evaluate()
    elif args.mode == 'balance':
        hierarchy.balance()
    else:
        print("Invalid mode. Please choose from 'build', 'evaluate', 'balance'.")
        

        


