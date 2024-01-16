import torch
import lightning.pytorch as pl


class TripletLoss(pl.LightningModule):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def _pairwise_distances(self, zis, zls, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                    If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(zls, zis.t())

        a_square_norm = torch.diag(torch.matmul(zls, zls.t()))
        b_square_norm = torch.diag(torch.matmul(zis, zis.t()))
        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        # square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2 # TODO: not a^2
        # shape (batch_size, batch_size)
        distances = a_square_norm.unsqueeze(0) - 2.0 * dot_product + b_square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances

    def get_valid_positive_mask(self, labels):
        """
        To be a valid positive pair (a,p),
            - a and p are different embeddings
            - a and p have the same label
        """
        indices_equal = torch.eye(labels.size(0), dtype=torch.int8, device=self.device)
        # indices_not_equal = ~indices_equal

        # label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

        return indices_equal

    def get_valid_negative_mask(self, labels):
        """
        To be a valid negative pair (a,n),
            - a and n are different embeddings
            - a and n have the different label
        """
        indices_equal = torch.eye(labels.size(0), dtype=torch.int8, device=self.device)
        indices_not_equal = torch.logical_not(indices_equal)

        # label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

        return indices_not_equal

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.int32, device=self.device)
        indices_not_equal = torch.logical_not(indices_equal)
        i_equal_j = torch.unsqueeze(indices_equal, 2)
        i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
        j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

        distinct_indices = torch.logical_and(torch.logical_and(i_equal_j, i_not_equal_k), j_not_equal_k)

        # # Check if labels[i] == labels[j] and labels[i] != labels[k]
        # label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        # i_equal_j = tf.expand_dims(label_equal, 2)
        # i_equal_k = tf.expand_dims(label_equal, 1)

        # valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # # Combine the two masks
        # mask = tf.logical_and(distinct_indices, valid_labels)

        return distinct_indices

    # def forward(self, zis, zls, squared=False):
    # def batch_hard_triplet_loss(self, zis, zls, squared=False):
    #     """Build the triplet loss over a batch of embeddings.
    #
    #     For each anchor, we get the hardest positive and hardest negative to form a triplet.
    #
    #     Args:
    #         labels: labels of the batch, of size (batch_size,)
    #         zis, zls: tensor of shape (batch_size, embed_dim)
    #         margin: margin for triplet loss
    #         squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
    #                 If false, output is the pairwise euclidean distance matrix.
    #
    #     Returns:
    #         triplet_loss: scalar tensor containing the triplet loss
    #     """
    #     batch_size = zis.size(0)
    #     labels = torch.arange(batch_size).to(self.device)
    #     # Get the pairwise distance matrix
    #     distances = self._pairwise_distances(zis, zls, squared=squared)
    #
    #     mask_positive = self.get_valid_positive_mask(labels)
    #     hardest_positive_dist = (distances * mask_positive.float()).max(dim=1)[0]
    #
    #     mask_negative = self.get_valid_negative_mask(labels)
    #     max_negative_dist = distances.max(dim=1, keepdim=True)[0]
    #     distances = distances + max_negative_dist * (~mask_negative).float()
    #     hardest_negative_dist = distances.min(dim=1)[0]
    #
    #     triplet_loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)
    #     triplet_loss = triplet_loss.mean()
    #
    #     return triplet_loss
    #
    # # def forward(self, zis, zls, squared=False):
    # def batch_all_triplet_loss(self, zis, zls, squared=False):
    #     """Build the triplet loss over a batch of embeddings.
    #     We generate all the valid triplets and average the loss over the positive ones.
    #     Args:
    #         labels: labels of the batch, of size (batch_size,)
    #         embeddings: tensor of shape (batch_size, embed_dim)
    #         margin: margin for triplet loss
    #         squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
    #                 If false, output is the pairwise euclidean distance matrix.
    #     Returns:
    #         triplet_loss: scalar tensor containing the triplet loss
    #     """
    #     batch_size = zis.size(0)
    #     labels = torch.arange(batch_size).to(self.device)
    #
    #     # Get the pairwise distance matrix
    #     distances = self._pairwise_distances(zis, zls, squared=squared)
    #
    #     # shape (batch_size, batch_size, 1)
    #     anchor_positive_dist = torch.unsqueeze(distances, 2)
    #     assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    #     # shape (batch_size, 1, batch_size)
    #     anchor_negative_dist = torch.unsqueeze(distances, 1)
    #     assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    #
    #     # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    #     # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    #     # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    #     # and the 2nd (batch_size, 1, batch_size)
    #     triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
    #
    #     # Put to zero the invalid triplets
    #     # (where label(a) != label(p) or label(n) == label(a) or a == p)
    #     mask = self._get_triplet_mask(labels)
    #     mask = mask.float().to(self.device)
    #     triplet_loss = torch.mul(mask, triplet_loss)
    #
    #     # Remove negative losses (i.e. the easy triplets)
    #     triplet_loss = torch.maximum(triplet_loss, torch.Tensor([0.0]).to(self.device))
    #
    #     # Count number of positive triplets (where triplet_loss > 0)
    #     valid_triplets = torch.gt(triplet_loss, 1e-16).float()
    #     num_positive_triplets = torch.sum(valid_triplets)
    #     num_valid_triplets = torch.sum(mask)
    #     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    #
    #     # Get final mean triplet loss over the positive valid triplets
    #     triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    #
    #     return triplet_loss

    # def forward(self, zis, zls):
    # def normal(self, zis, zls):
    #     distances = self._pairwise_distances(zis, zls)
    #     loss_list = []
    #     for i in range(self.batch_size):
    #         for j in range(self.batch_size):
    #             if i == j:
    #                 continue
    #             loss_list.append(max(0, distances[i][i] - distances[i][j]))
    #
    #     loss = sum(loss_list)
    #
    #     return loss

    def forward(self, zis, zls):
        batch_size = zis.shape[0]
        distances = self._pairwise_distances(zis, zls)
        loss_list = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                if distances[i][i] < distances[i][j] < distances[i][i] + self.margin:  # semi-hard
                    loss_list.append(distances[i][i] - distances[i][j] + self.margin)

        if len(loss_list) == 0:  # margin is set to a too small value
            print("loss_list is 0")
            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j:
                        continue
                    if distances[i][j] < distances[i][i]:  # hard
                        loss_list.append(distances[i][i] - distances[i][j] + self.margin)

        loss = sum(loss_list) / len(loss_list)

        return loss
