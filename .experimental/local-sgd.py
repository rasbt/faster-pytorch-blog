import os
import os.path as op
import time



from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from watermark import watermark
from accelerate import Accelerator

from local_dataset_utilities import (
    download_dataset,
    load_dataset_into_to_dataframe,
    partition_dataset,
)
from local_dataset_utilities import IMDBDataset


class LocalSGD:
    """
      A helper class to support local SGD on top of Accelerator.
      It simply runs a given number of updates independently on each device,
      and averages model weights every K synchronization step.
      Contributed by Leo (Leonid) Boytsov.
      Although we are not aware of the true origins of this simple approach,
      the idea of local SGD is quite old and goes back to at least:
      Zhang, J., De Sa, C., Mitliagkas, I., & Ré, C. (2016). 
      Parallel SGD: When does averaging help?. arXiv preprint arXiv:1606.07365.
      We credit the term Local SGD to the following paper (but there might be
      earlier references we are not aware of).
      Stich, Sebastian Urban. "Local SGD Converges Fast and Communicates Little." 
      ICLR 2019-International Conference on Learning Representations. No. CONF. 2019.
    """

    def __enter__(self):
        if self.enabled:
            self.model_sync_obj = self.model.no_sync()
            self.model_sync_obj.__enter__()

        return self

    def __exit__(self, type, value, tb):
        if self.enabled:
            # Average all models on exit
            self._sync_and_avg_model_params()
            self.model_sync_obj.__exit__(type, value, tb)

    def __init__(self, accelerator: Accelerator, model: torch.nn.Module,
                 local_sgd_steps: int, enabled: bool = True):
        """
            Constructor.
            Args:
                model (`torch.nn.Module):
                    The model whose parameters we need to average.
                accelerator (`Accelerator`):
                    Accelerator object.
                local_sgd_steps (`int`):
                    A number of local SGD steps (before model parameters are synchronized).
                enabled (`bool):
                    Local SGD is disabled if this parameter set to `False`.
        """
        self.enabled = enabled
        self.step_qty = 0
        if self.enabled:
            self.accelerator = accelerator
            self.model = model
            self.local_sgd_steps = local_sgd_steps

    def step(self):
        """
            This function makes a "step" and synchronizes model parameters if necessary.
        """
        self.step_qty += 1
        if not self.enabled:
            return

        if self.step_qty % self.local_sgd_steps == 0:
            self._sync_and_avg_model_params()

    def _sync_and_avg_model_params(self):
        """
            Synchronize + Average model parameters across all GPUs
        """
        import torch.distributed as dist
        self.accelerator.wait_for_everyone()
        with self.accelerator.autocast():
            qty = float(dist.get_world_size())
            for prm in self.model.parameters():
                dist.all_reduce(prm.data, op=torch.distributed.ReduceOp.SUM)
                prm.data /= qty

def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


def train(num_epochs, model, optimizer, train_loader, val_loader, device):

    local_sgd_steps=8


    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

        with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
            for batch_idx, batch in enumerate(train_loader):
                model.train()
                for s in ["input_ids", "attention_mask", "label"]:
                    batch[s] = batch[s].to(device)

                ### FORWARD AND BACK PROP
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"],
                )
                optimizer.zero_grad()
                local_sgd.step()
                accelerator.backward(outputs["loss"])

                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                ### LOGGING
                if not batch_idx % 300:
                    print(
                        f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {outputs['loss']:.4f}"
                    )

                model.eval()
                with torch.no_grad():
                    predicted_labels = torch.argmax(outputs["logits"], 1)
                    train_acc.update(predicted_labels, batch["label"])

        ### MORE LOGGING
        with torch.no_grad():
            model.eval()
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
            for batch in val_loader:
                for s in ["input_ids", "attention_mask", "label"]:
                    batch[s] = batch[s].to(device)
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"],
                )
                predicted_labels = torch.argmax(outputs["logits"], 1)
                val_acc.update(predicted_labels, batch["label"])

            print(
                f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%"
            )


if __name__ == "__main__":
    print(watermark(packages="torch,lightning,transformers", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(123)

    ##########################
    ### 1 Loading the Dataset
    ##########################
    download_dataset()
    df = load_dataset_into_to_dataframe()
    if not (op.exists("train.csv") and op.exists("val.csv") and op.exists("test.csv")):
        partition_dataset(df)

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    #########################################
    ### 2 Tokenization and Numericalization
    #########################################

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizer input max length:", tokenizer.model_max_length, flush=True)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size, flush=True)

    print("Tokenizing ...", flush=True)
    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    del imdb_dataset
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #########################################
    ### 3 Set Up DataLoaders
    #########################################

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=2,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=2,
        drop_last=True,
    )

    #########################################
    ### 4 Initializing the Model
    #########################################

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    #########################################
    ### 5 Finetuning
    #########################################

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)

    start = time.time()
    train(
        num_epochs=3,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
        for batch in test_loader:
            for s in ["input_ids", "attention_mask", "label"]:
                batch[s] = batch[s].to(device)
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            predicted_labels = torch.argmax(outputs["logits"], 1)
            test_acc.update(predicted_labels, batch["label"])

    print(f"Test accuracy {test_acc.compute()*100:.2f}%")