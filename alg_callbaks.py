from CONSTANTS import *


class ALGCallback(Callback):

    def on_init_start(self, trainer):
        print('--- Starting to init trainer! ---')

    def on_init_end(self, trainer):
        print('--- trainer is init now ---')

    def on_train_end(self, trainer, pl_module):
        # print('--- training ends ---')
        if SAVE_WEIGHTS:
            torch.save(pl_module.net.state_dict(), "example.ckpt")

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """Called when the train epoch ends."""
        pass
