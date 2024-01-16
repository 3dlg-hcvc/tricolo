from tricolo.data.dataset.general_dataset import GeneralDataset


class Text2ShapeChairTable(GeneralDataset):
    def __init__(self, cfg, split):
        super(Text2ShapeChairTable, self).__init__(cfg, split)
