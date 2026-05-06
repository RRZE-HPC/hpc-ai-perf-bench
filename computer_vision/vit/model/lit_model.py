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
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(image_size=img_size)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(image_size=img_size)
        elif model_name == "vit_h_14":
            self.model = models.vit_h_14(image_size=img_size)
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
    model = LitModel("vit_b_16", img_size=image_size)
    x = torch.randn(batch_size, 3, image_size, image_size)
    y = torch.randint(0, 1000, (batch_size,))
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()
    model(x)
    batch = {"img": x, "label": y}
    model.training_step(batch, 0)