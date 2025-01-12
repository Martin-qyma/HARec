import torch
import pickle
from config import args
from typing import List
from manifolds.hyperboloid import Hyperboloid
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        return self.input_text[idx]


class DataHandler:
    def __init__(self):
        self.manifold = Hyperboloid()
        if args.dataset == "amazon":
            self.user_prompt = "Describe what types of books this user is likely to enjoy within 50 words."
            self.item_prompt = "Describe what types of users would enjoy this book within 50 words."
            self.item = "book"
        elif args.dataset == "yelp" or args.dataset == "google":
            self.user_prompt = "Describe what types of business this user is likely to enjoy within 50 words."
            self.item_prompt = "Describe what types of users would enjoy this business within 50 words."
            self.item = "business"

        self.user_emb = pickle.load(open(f"../data/{args.dataset}/user_embed.pkl", "rb")).detach()
        self.item_emb = pickle.load(open(f"../data/{args.dataset}/item_embed.pkl", "rb")).detach()
        
        # map the embeddings to the euclidean space
        self.user_emb = self.manifold.expmap0(self.user_emb, c=torch.tensor([1.0]).cuda())
        self.item_emb = self.manifold.expmap0(self.item_emb, c=torch.tensor([1.0]).cuda())
        

    def load_data(self):
        with open(f"../data/{args.dataset}/usr_prf.pkl", "rb") as file:
            usr_prf = pickle.load(file)
        with open(f"../data/{args.dataset}/itm_prf.pkl", "rb") as file:
            itm_prf = pickle.load(file)
        
        # usr_prf is a dictionary, separate it into training, validation, and testing set
        usr_trn_dict = {}
        usr_val_dict = {}
        usr_tst_dict = {}
        itm_trn_dict = {}
        itm_val_dict = {}
        itm_tst_dict = {}
        for key in usr_prf.keys():
            if key % 10 == 0:
                usr_val_dict[key] = usr_prf[key]
            elif key % 10 == 1:
                usr_tst_dict[key] = usr_prf[key]
            else:
                usr_trn_dict[key] = usr_prf[key]
        for key in itm_prf.keys():
            if key % 10 == 0:
                itm_val_dict[key] = itm_prf[key]
            elif key % 10 == 1:
                itm_tst_dict[key] = itm_prf[key]
            else:
                itm_trn_dict[key] = itm_prf[key]

        # combine all information input input string
        trn_input = []
        val_input = []
        tst_input = []
        for uid in usr_trn_dict.keys():
            user_message = f"<USER_EMBED> {usr_trn_dict[uid]['profile']}"

            trn_input.append(
                (
                    "user",
                    self.user_emb[uid],
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.user_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
            )
        for iid in itm_trn_dict.keys():
            item_message = f"<ITEM_EMBED> {itm_trn_dict[iid]['profile']}"
            trn_input.append(
                (
                    "item",
                    self.item_emb[iid],
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.item_prompt}<|eot_id|><|start_header_id|>item<|end_header_id|>{item_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
            )
        for uid in usr_val_dict.keys():
            user_message = f"<USER_EMBED>"
            val_input.append(
                (
                    "user",
                    self.user_emb[uid],
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.user_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    usr_val_dict[uid]['profile']
                )
            )
        for iid in itm_val_dict.keys():
            item_message = f"<ITEM_EMBED>"
            val_input.append(
                (
                    "item",
                    self.item_emb[iid],
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.item_prompt}<|eot_id|><|start_header_id|>item<|end_header_id|>{item_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    itm_val_dict[iid]['profile']
                )
            )
        for uid in usr_tst_dict.keys():
            user_message = f"<USER_EMBED>"
            tst_input.append(
                (
                    "user",
                    self.user_emb[uid],
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.user_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    usr_tst_dict[uid]['profile']
                )
            )
        for iid in itm_tst_dict.keys():
            item_message = f"<ITEM_EMBED>"
            tst_input.append(
                (
                    "item",
                    self.item_emb[iid],
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.item_prompt}<|eot_id|><|start_header_id|>item<|end_header_id|>{item_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    itm_tst_dict[iid]['profile']
                )
            )

        # load training batch
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # load testing batch
        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True)

        return trn_loader, val_loader, tst_loader