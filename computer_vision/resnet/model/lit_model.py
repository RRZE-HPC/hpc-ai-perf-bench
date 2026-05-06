import torch
from torch.nn import functional as F
from torchvision import models
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self, model_name, img_size=224, batch_size=None):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        if model_name == "resnet18":
            self.model = models.resnet18()
        elif model_name == "resnet50":
            self.model = models.resnet50()
        elif model_name == "resnet101":
            self.model = models.resnet101()
        else:
            raise ValueError(f"Unknown model: {model_name}")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == "__main__":
    batch_size = 16
    image_size = 384
    model = LitModel("resnet18")
    x = torch.randn(batch_size, 3, image_size, image_size)
    y = torch.randint(0, 1000, (batch_size,))
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()
    model(x)
    batch = {"img": x, "label": y}
    model.training_step(batch, 0)
    print("training step done")