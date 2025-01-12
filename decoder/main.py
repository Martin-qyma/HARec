import os
import json
import torch
import pickle
import torch.nn as nn
from config import args
from decoder import Decoder
from evaluater import MetricScore
from utils.data_handler import DataHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Describe:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        self.model = Decoder().to(device)
        self.data_handler = DataHandler()
        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()

        self.user_embedding_converter_path = f"../data/{args.dataset}/user_converter.pkl"
        self.item_embedding_converter_path = f"../data/{args.dataset}/item_converter.pkl"
        self.tst_predictions_path = f"../data/{args.dataset}/profile_pred.pkl"
        self.tst_references_path = f"../data/{args.dataset}/profile_ref.pkl"

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            total_loss = 0
            self.model.train()
            for i, batch in enumerate(self.trn_loader):
                name, embed, input_text = batch
                embed = embed.to(device)

                input_ids, outputs = self.model.forward(name[0], embed, input_text[0])
                input_ids = input_ids.to(device)
                optimizer.zero_grad()
                loss = self.model.loss(input_ids, outputs, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 1000 == 0 and i != 0:
                    print(f"Epoch [{epoch}/{args.epochs}], Batch [{i}/{len(self.trn_loader)}], Loss: {loss.item()}")
                    output_text = self.model.generate(name[0], embed, input_text[0])
                    print(f"Generated Description: {output_text}") 

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {total_loss}")

            # Save the model
            torch.save(
                self.model.user_embedding_converter.state_dict(),
                self.user_embedding_converter_path,
            )
            torch.save(
                self.model.item_embedding_converter.state_dict(),
                self.item_embedding_converter_path,
            )
            print(f"Saved model to {self.user_embedding_converter_path}")
            print(f"Saved model to {self.item_embedding_converter_path}")

    def generate(self):
        loader = self.tst_loader

        # load model
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                name, embed, input_text, groudtruth = batch
                embed = embed.to(device)
                outputs = self.model.generate(name[0], embed, input_text[0])
                predictions.append(outputs[0])
                references.append(groudtruth[0])
                if i % 100 == 0 and i != 0:
                    print(f"Step [{i}/{len(loader)}]")
                    print(f"Generated Description: {outputs[0]}")
                    print(f"Groundtruth Description: {groudtruth[0]}")
                if i == 1000:
                    break

        with open(self.tst_predictions_path, "wb") as file:
            pickle.dump(predictions, file)
        with open(self.tst_references_path, "wb") as file:
            pickle.dump(references, file)
        print(f"Saved predictions to {self.tst_predictions_path}")
        print(f"Saved references to {self.tst_references_path}")   

def main():
    sample = Describe()
    if args.mode == "finetune":
        print("Finetuning model...")
        sample.train()
    elif args.mode == "generate":
        print("Generating descriptions...")
        sample.generate()
    elif args.mode == "evaluate":
        print("Evaluating model...")
        metric_score = MetricScore()
        metric_score.print_score()
    else:
        print("Invalid mode. Please choose 'finetune', 'generate', or 'evaluate'.")

if __name__ == "__main__":
    main()