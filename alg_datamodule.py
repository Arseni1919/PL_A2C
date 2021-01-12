from typing import Union, List, Any

from CONSTANTS import *
from alg_dataset import ALGDataset


class ALGDataModule(pl.LightningDataModule):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.env = gym.make(ENV)

    def prepare_data(self):
        state = self.env.reset()
        for i in range(WARM_START_STEPS):
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            experience = Experience(state, action, reward, done, new_state)
            self.dataset.append(experience)
            state = self.env.reset() if done else new_state
        print('--- finished prepare_data ---')

    def setup(self, stage=None):
        # transforms
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=BATCH_SIZE)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass


