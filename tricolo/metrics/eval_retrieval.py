"""
Modifed code from: https://github.com/kchen92/text2shape/blob/master/tools/eval/eval_text_encoder.py
"""

import argparse
import collections
import datetime
import json
import numpy as np
import os
import pickle

def construct_embeddings_matrix(dataset, embeddings_dict, shape_embedding_input, model_id_to_label=None, label_to_model_id=None):
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
    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][3]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = len(embeddings_dict['caption_embedding_tuples'])
    if (dataset == 'shapenet') and (num_embeddings > 30000):
        raise ValueError('Too many ({}) embeddings. Only use up to 30000.'.format(num_embeddings))
    assert embedding_sample.ndim == 1

    # Print info about embeddings
    print('Number of embeddings:', num_embeddings)
    print('Dimensionality of embedding:', embedding_dim)
    print('Estimated size of embedding matrix (GB):',
          embedding_dim * num_embeddings * 4 / 1024 / 1024 / 1024)
    print('')

    # Create embeddings matrix (n_samples x n_features) and vector of labels
    if shape_embedding_input:
        embeddings_matrix = []
        labels = []
    else:
        embeddings_matrix = np.zeros((num_embeddings, embedding_dim))
        labels = np.zeros((num_embeddings)).astype(int)

    if (model_id_to_label is None) and (label_to_model_id is None):
        model_id_to_label = {}
        label_to_model_id = {}
        label_counter = 0
        new_dicts = True
    else:
        new_dicts = False

    model_id_set = []
    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):

        if shape_embedding_input:
            caption, category, model_id, text_embedding, shape_embedding = caption_tuple
            embedding = shape_embedding
        else:
            caption, category, model_id, text_embedding, shape_embedding = caption_tuple
            embedding = text_embedding

        if dataset == 'primitives':
            if new_dicts:
                # For primitives modelids under the same category share the same text descriptions
                if category not in model_id_to_label:
                    model_id_to_label[category] = label_counter
                    label_to_model_id[label_counter] = category
                    label_counter += 1

            if shape_embedding_input:
                # Each different modelid should be included in the embedding matrix, but they belong to the same label category
                if model_id not in model_id_set:
                    embeddings_matrix.append(embedding)
                    labels.append(model_id_to_label[category])
            else:
                # All descriptions should be included
                embeddings_matrix[idx] = embedding
                labels[idx] = model_id_to_label[category]

            model_id_set.append(model_id)
            model_id_set = list(set(model_id_set))
        else:
            if new_dicts:
                if model_id not in model_id_to_label:
                    model_id_to_label[model_id] = label_counter
                    label_to_model_id[label_counter] = model_id
                    label_counter += 1

            if shape_embedding_input:
                if model_id not in model_id_set:
                    embeddings_matrix.append(embedding)
                    labels.append(model_id_to_label[model_id])
            else:
                embeddings_matrix[idx] = embedding
                labels[idx] = model_id_to_label[model_id]

            model_id_set.append(model_id)
            model_id_set = list(set(model_id_set))

        # Print progress
        if (idx + 1) % 10000 == 0:
            print('Processed {} / {} embeddings'.format(idx + 1, num_embeddings))
        
    if shape_embedding_input:
        embeddings_matrix = np.vstack(embeddings_matrix)
        labels = np.array(labels).astype(int)
    return embeddings_matrix, labels, model_id_to_label, num_embeddings, label_to_model_id

def _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query, range_start=0):
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)

    print(unnormalized_similarities.shape)

    n_samples = unnormalized_similarities.shape[0]
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    indices = sort_indices[:, -n_neighbors:]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)]
    yo = unnormalized_similarities[row_indices, indices.flatten()].reshape(n_samples, n_neighbors)
    indices = indices[row_indices, np.argsort(yo, axis=1).flatten()].reshape(n_samples, n_neighbors)
    indices = np.flip(indices, 1)
    return indices

def compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query):
    print('Using unnormalized cosine distance')
    n_samples = query_embeddings_matrix.shape[0]
    return None, _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query)

def compute_nearest_neighbors(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, metric='cosine'):
    fit_eq_query = False
    if metric == 'cosine':
        distances, indices = compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, n_neighbors, fit_eq_query)
    else:
        raise ValueError('Use cosine distance.')
    return distances, indices

def compute_pr_at_k(embeddings_path, indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)
    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """
    fo = open(os.path.join(embeddings_path, "pr_at_k.txt"), 'w')

    assert fit_labels is not None
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    # Assumes that self is not included in the nearest neighbors
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

    matrix = num_correct/num_relevant[:,None]
    nan_idx = np.argwhere(np.isnan(num_correct/num_relevant[:,None]))

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.cumsum(dcg_n/dcg_d,axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal/dcg_d,axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings

    # Placeholder for now
    # r_rank = 0

    fo.write('     k: precision recall recall_rate ndcg\n')
    for k in range(n_neighbors):
        # fo.write('pr @ {}: {} {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k], r_rank) + '\n')
        fo.write('pr @ {}: {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k]) + '\n')

    Metrics = {}
    Metrics['precision'] = precision_at_k
    Metrics['recall'] = recall_at_k
    Metrics['recall_rate'] = recall_rate_at_k
    Metrics['ndcg'] = ave_ndcg_at_k
    # Metrics['r_rank'] = r_rank
    return Metrics

def compute_cross_modal(dataset, embeddings_dict, embeddings_path, metric='cosine', concise=False):
    (text_embeddings_matrix, text_labels, model_id_to_label, text_num_embeddings, label_to_model_id) = construct_embeddings_matrix(dataset, embeddings_dict, shape_embedding_input=False)
    (shape_embeddings_matrix, shape_labels, shape_model_id_to_label, shape_num_embeddings, shape_label_to_model_id) = construct_embeddings_matrix(dataset, embeddings_dict, 
                                                                                                                                                  shape_embedding_input=True,
                                                                                                                                                  model_id_to_label=model_id_to_label,
                                                                                                                                                  label_to_model_id=label_to_model_id)

    assert shape_model_id_to_label == model_id_to_label
    assert shape_label_to_model_id == label_to_model_id

    n_neighbors = 20
    distances, indices = compute_nearest_neighbors(shape_embeddings_matrix, text_embeddings_matrix, n_neighbors, metric)
    pr_at_k = compute_pr_at_k(embeddings_path, indices, text_labels, n_neighbors, num_embeddings=indices.shape[0], fit_labels=shape_labels)
    return pr_at_k
