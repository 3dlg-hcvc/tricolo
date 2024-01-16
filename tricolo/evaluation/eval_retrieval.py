import numpy as np
import os
import jsonlines


def construct_embeddings_matrix(dataset, embeddings_dict, model_id_to_label=None, label_to_model_id=None):
    """Construct the embeddings matrix, which is NxD where N is the number of embeddings and D is
    the dimensionality of each embedding.
    Args:
        dataset: String specifying the dataset (e.g. 'synthetic' or 'shapenet')
        embeddings_dict: Dictionary containing the embeddings. It should have keys such as
                the following: ['caption_embedding_tuples', 'dataset_size'].
                caption_embedding_tuples is a list of tuples where each tuple can be decoded like
                so: caption, category, model_id, embedding = caption_tuple.
    """
    assert (((model_id_to_label is None) and (label_to_model_id is None)) or
            ((model_id_to_label is not None) and (label_to_model_id is not None)))
    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][-1]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = len(embeddings_dict['caption_embedding_tuples'])  # 7424

    assert embedding_sample.ndim == 1

    # Create embeddings matrix (n_samples x n_features) and vector of labels
    text_embeddings_matrix = np.zeros((num_embeddings, embedding_dim))
    shape_embeddings_list = []
    labels = np.zeros(shape=num_embeddings, dtype=np.int64)
    labels_shape = []

    if (model_id_to_label is None) and (label_to_model_id is None):
        model_id_to_label = {}
        label_to_model_id = {}
        label_counter = 0
        new_dicts = True
    else:
        new_dicts = False

    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):

        # Parse caption tuple
        caption, category, model_id, text_embedding, shape_embedding = caption_tuple

        # Swap model ID and category depending on dataset
        # TODO: remove hard code
        if dataset == "Primitives":
            model_id = category

        # Add model ID to dict if it has not already been added
        if new_dicts:
            if model_id not in model_id_to_label:
                model_id_to_label[model_id] = label_counter
                label_to_model_id[label_counter] = model_id

                shape_embeddings_list.append(shape_embedding)
                labels_shape.append(label_counter)
                label_counter += 1

        # Update the embeddings matrix and labels vector
        text_embeddings_matrix[idx] = text_embedding
        labels[idx] = model_id_to_label[model_id]

    shape_embeddings_matrix = np.vstack(shape_embeddings_list)  # 1486,512
    labels_shape = np.array(labels_shape).astype(int)  # 1486

    return text_embeddings_matrix, shape_embeddings_matrix, labels, labels_shape, model_id_to_label, num_embeddings, label_to_model_id


def _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query,
                                      range_start=0):
    if fit_eq_query:
        n_neighbors += 1

    # Argsort method
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    sort_indices = np.argsort(unnormalized_similarities, axis=1)
    sort_distances = np.sort(unnormalized_similarities, axis=1)
    distances = sort_distances[:, -n_neighbors:]
    distances = np.flip(distances)
    # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    indices = sort_indices[:, -n_neighbors:]
    indices = np.flip(indices, 1)
    sort_indices = np.flip(sort_indices, 1)

    if fit_eq_query:
        n_neighbors -= 1  # Undo the neighbor increment
        final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
        compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
        has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
        any_result = np.any(has_self, axis=1)
        for row_idx in range(indices.shape[0]):
            if any_result[row_idx]:
                nonzero_idx = np.nonzero(has_self[row_idx, :])
                assert len(nonzero_idx) == 1
                new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
                final_indices[row_idx, :] = new_row
            else:
                final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
        indices = final_indices
    return distances, indices, sort_indices


def compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query):
    # print('Using normalized cosine distance')
    n_samples = query_embeddings_matrix.shape[0]
    if n_samples > 8000:  # Divide into blocks and execute
        def block_generator(mat, block_size):
            for i in range(0, mat.shape[0], block_size):
                yield mat[i:(i + block_size), :]

        block_size = 3000
        blocks = block_generator(query_embeddings_matrix, block_size)
        indices_list = []
        distances_list, sort_indices_list = [], []
        for cur_block_idx, block in enumerate(blocks):
            cur_distances, cur_indices, cur_sort_indices = _compute_nearest_neighbors_cosine(fit_embeddings_matrix,
                                                                                             block,
                                                                                             n_neighbors, fit_eq_query,
                                                                                             range_start=cur_block_idx * block_size)
            indices_list.append(cur_indices)
            distances_list.append(cur_distances)
            sort_indices_list.append(cur_sort_indices)
        indices = np.vstack(indices_list)
        distances = np.vstack(distances_list)
        sort_indices = np.vstack(sort_indices_list)
        return distances, indices, sort_indices
    else:
        distances, indices, sort_indices = _compute_nearest_neighbors_cosine(fit_embeddings_matrix,
                                                                             query_embeddings_matrix, n_neighbors,
                                                                             fit_eq_query)
        return distances, indices, sort_indices


def compute_nearest_neighbors(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors):
    """Compute nearest neighbors.
    Args:
        fit_embeddings_matrix: NxD matrix
    """
    fit_eq_query = False
    if (fit_embeddings_matrix.shape == query_embeddings_matrix.shape) and np.allclose(fit_embeddings_matrix, query_embeddings_matrix):
        fit_eq_query = True

    distances, indices, sort_indices = compute_nearest_neighbors_cosine(
        fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query
    )

    return distances, indices, sort_indices


def compute_pr_at_k(indices, sort_indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)
    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """

    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors), dtype=np.float32)
    rel_score = np.zeros((num_embeddings, n_neighbors), dtype=np.float32)
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors), dtype=np.float32)

    # Assumes that self is not included in the nearest neighbors
    r_rank = 0
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        # for now binary relevance
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k goes from 0 to n_neighbors-1
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

        # r_rank
        all_nearest = sort_indices[i]
        all_nearest_classes = [fit_labels[x] for x in all_nearest]
        r_rank += 1 / (all_nearest_classes.index(label) + 1)
    r_rank = r_rank / num_embeddings

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1, n_neighbors + 1) + 1)
    dcg = np.cumsum(dcg_n / dcg_d, axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal / dcg_d, axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct / num_relevant[:, None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct / np.arange(1, n_neighbors + 1), axis=0) / num_embeddings

    metrics = {
        'precision': precision_at_k, 'recall': recall_at_k, 'recall_rate': recall_rate_at_k,
        'ndcg': ave_ndcg_at_k, 'mrr': r_rank
    }
    return metrics


def get_nearest_info(indices, sort_indices, fit_labels, labels, label_to_model_id, caption_tuples):
    """Compute and return the model IDs of the nearest neighbors.
    """
    # r_rank
    r_rank_list = []
    for i in range(len(sort_indices)):
        label = labels[i]
        all_nearest = sort_indices[i]
        all_nearest_classes = [fit_labels[x] for x in all_nearest]
        r_rank = 1 / (all_nearest_classes.index(label) + 1)
        r_rank_list.append(r_rank)

    # Convert labels to model IDs
    query_model_ids = []
    cat_ids = []
    # query_sentences = []
    for idx, label in enumerate(labels):
        # query_model_ids.append(label_to_model_id[label])
        query_model_ids.append(caption_tuples[idx][2])
        cat_ids.append(caption_tuples[idx][1])
        # cur_sentence_as_word_indices = caption_tuples[idx][0]
        # if cur_sentence_as_word_indices is None:
        #     query_sentences.append('None (shape embedding)')
        # else:
        #     query_sentences.append(' '.join([idx_to_word[str(word_idx)]
        #                                     for word_idx in cur_sentence_as_word_indices
        #                                     if word_idx != 0]))

    # Convert neighbors to model IDs
    nearest_model_ids = []
    for row in indices:
        model_ids = []
        for col in row:
            model_ids.append(label_to_model_id[col])
        nearest_model_ids.append(model_ids)
    assert len(query_model_ids) == len(nearest_model_ids)
    return query_model_ids, cat_ids, nearest_model_ids, r_rank_list


def compute_metrics(dataset, embeddings_dict, print_results=False):
    """
    Compute all the metrics for the text encoder evaluation.
    """

    (text_embeddings_matrix, shape_embeddings_matrix, labels, fit_labels, model_id_to_label,
     num_embeddings, label_to_model_id) = construct_embeddings_matrix(dataset, embeddings_dict)

    n_neighbors = 5  # 20 change

    distances, indices, sort_indices = compute_nearest_neighbors(
        shape_embeddings_matrix, text_embeddings_matrix, n_neighbors
    )  # change text_em and shape_em

    pr_at_k = compute_pr_at_k(indices, sort_indices, labels, n_neighbors, num_embeddings, fit_labels)

    query_model_ids, cat_ids, nearest_model_ids, r_ranks = get_nearest_info(
        indices,
        sort_indices,
        fit_labels,
        labels,
        label_to_model_id,
        embeddings_dict['caption_embedding_tuples'],
    )  # change labels and fit_labels

    print_nearest_info(cat_ids, query_model_ids, nearest_model_ids, distances)

    if print_results:
        _print_results(pr_at_k)
    return pr_at_k


def print_nearest_info(categories, query_model_ids, nearest_model_ids, distances):
    """Print out nearest model IDs for random queries.
    Args:
        labels: 1D array containing the label
    """
    # Make directory for renders


    perm = np.random.permutation(len(nearest_model_ids))
    # cnt = 0

    jsonl_writer = jsonlines.open('nearest.jsonl', mode='w')

    for i in perm:  # TODO:
        # for i in range(len(nearest_model_ids)):
        query_model_id = query_model_ids[i]
        cat_id = categories[i]
        distance = distances[i]
        nearest = nearest_model_ids[i]


        jsonl_writer.write(
            {'cat_id': cat_id, 'groundtruth': query_model_id + ('-%04d' % i), 'retrieved_models': nearest,
             'distance': distance.tolist()})




def _print_results(pr_at_k):
    print("\nRR@1 RR@5 NDCG@5 MRR")
    print(
        f'{round(pr_at_k["recall_rate"][0] * 100, 2)} {round(pr_at_k["recall_rate"][4] * 100, 2)} {round(pr_at_k["ndcg"][4] * 100, 2)} {round(pr_at_k["mrr"] * 100, 2)}'
    )
