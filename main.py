from CONSTANTS import *
from alg_lightning_module import ALGLightningModule
from alg_datamodule import ALGDataModule
from alg_callbaks import ALGCallback
from alg_dataset import ALGDataset
from try_weights import play


def main():
    dataset = ALGDataset()
    model = ALGLightningModule(dataset)
    data_module = ALGDataModule(dataset)

    trainer = pl.Trainer(callbacks=[ALGCallback()], max_epochs=MAX_EPOCHS)
    trainer.fit(model=model, datamodule=data_module)

    play(NUMBER_OF_GAMES)


if __name__ == '__main__':
    main()

    # to run tensorboard:
    # tensorboard --logdir lightning_logs

