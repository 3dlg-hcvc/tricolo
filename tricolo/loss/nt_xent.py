import torch
import lightning.pytorch as pl
import torch.nn.functional as F


class NTXentLoss(pl.LightningModule):
    """
    This NTXentLoss implementation is taken from: https://github.com/edreisMD/ConVIRT-pytorch/blob/master/loss/nt_xent.py
    """
    def __init__(self, temperature, alpha_weight):
        super().__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight

    def _softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=True):

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        labels = torch.eye(batch_size, device=self.device, dtype=torch.float32)

        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2, 0, 1)) / self.temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1, 0, 1)) / self.temperature

        loss_a = self._softXEnt(labels, logits_ab)
        loss_b = self._softXEnt(labels, logits_ba)

        return self.alpha_weight * loss_a + (1 - self.alpha_weight) * loss_b
