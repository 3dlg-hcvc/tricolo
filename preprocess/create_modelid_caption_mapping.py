import os
import json
import pickle
import argparse
import jsonlines

def prepare_json(root_dir, dataset, split):
    '''
    Read processed_captions_split.p and find matching modelid-caption pairs. Write to json file.
    Parameters
    ----------------
    root_dir: str
        File path of processed_captions_split.p
    dataset: str
        shapenet/primitives
    split: str
        train/val/test
    
    Returns
    ---------------
    json file named split_map.json
    '''
    pickle_path = os.path.join(root_dir, f'processed_captions_{split}.p')
    print('pickle path: {}'.format(pickle_path))
    with open(pickle_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    caption_tuples = embeddings_dict["caption_tuples"]
    # caption_tuples: list, (array, category, nrrd name)

    json_path = os.path.join('../datasets/text2shape-data/', dataset, dataset+'.json')
    print('json path: {}'.format(json_path))
    with open(json_path, 'r') as f:
        primitives = json.load(f)
    idx2word = primitives['idx_to_word']
    word2idx = primitives['word_to_idx']

    samples = []
    for inds, category, nrrd_name in caption_tuples:
        text = []
        for ind in inds:
            if ind==0: # 0: pad
                break
            text.append(idx2word[str(ind)])
        text = ' '.join(text)
        split_name = nrrd_name.split('.')[0]
        samples.append({'model': split_name, 'category': category,'caption': text, 'arrays': inds.tolist()})
    print('{} samples in {}'.format(len(samples), split))
    # primitives: train: 1524430 # val 177510 # test 216560
    # shapenet: train: 59777 # val 7435 # test 7452
    
    dest = os.path.join(root_dir, f'{split}_map.jsonl')
    print('Saving results to: {}'.format(dest))
    with jsonlines.open(dest, 'w') as writer:
        writer.write_all(samples)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shapenet", help="Dataset name (shapenet or primitives)")
    args = parser.parse_args()

    dataset = args.dataset # change
    root_dir = '../datasets/text2shape-data/'+dataset

    print('Generating jsonl for train')
    prepare_json(root_dir, dataset, 'train')
    print('Done processing for train')
    print()

    print('Generating jsonl for val')
    prepare_json(root_dir, dataset, 'val')
    print('Done processing for val')
    print()

    print('Generating jsonl for test')
    prepare_json(root_dir, dataset, 'test')
    print('Done processing for test')