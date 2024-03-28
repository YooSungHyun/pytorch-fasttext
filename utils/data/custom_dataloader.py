import torch
from torch.nn.utils.rnn import pad_sequence


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        # make pad or something work for each step's batch
        input_ids = list()
        labels = list()
        negative_samples = list()

        for i in range(len(batch)):
            input_ids.append(torch.LongTensor(batch[i]["input_ids"][0]))
            labels.append(torch.LongTensor(batch[i]["labels"]))
            negative_samples.append(torch.LongTensor(batch[i]["negative_samples"]))

        # if you neeed to many inputs, plz change this line
        # TODO(User): `inputs` must match the input argument of the model exactly (the current example only utilizes `inputs`).
        input_ids = pad_sequence(input_ids, batch_first=True)

        _returns = {
            "input_ids": input_ids,
            "labels": torch.concat(labels),
            "negative_samples": torch.concat(negative_samples),
        }
        # _returns = {"input_ids", "attention_mask", "input_type_ids", "labels"}

        return _returns
