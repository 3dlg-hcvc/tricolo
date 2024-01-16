from tricolo.data.dataset.general_dataset import GeneralDataset


class Text2ShapeC13(GeneralDataset):
    def __init__(self, cfg, split):
        super(Text2ShapeC13, self).__init__(cfg, split)
