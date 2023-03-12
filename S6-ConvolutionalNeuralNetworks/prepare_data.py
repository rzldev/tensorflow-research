## Preparing Data Sample From Food 101 ##

import os
import json
import shutil

def prepare(path, output_path, datasets=['train', 'test'], labels=['pizza', 'steak'],
            meta_dir='meta'):
    '''
    Preparing dataset directory to use to train CNN model
    '''
    print('\nPreparing data.')
    for dataset in datasets:
        copy_images(path, output_path, dataset, meta_dir, labels)
        
    for label in labels:
        train_files = set(os.listdir(f'{output_path}/train/{label}'))
        test_files = set(os.listdir(f'{output_path}/test/{label}'))
        
        assert len(train_files.intersection(test_files)) == 0
        assert len(train_files.intersection(test_files)) == 0

def copy_images(path, output_path, dataset, meta_dir, target_labels):
    '''
    Copy images from initial dir path to the new subset dir
    '''
    print(f'\nUsing {dataset} labels...')
    labels = get_labels(f'{path}', meta_dir, filename=dataset + '.json')
    print(f'Got {len(labels)} class names for {dataset} labels')

    if len(list(set(target_labels) & set(labels))) != len(target_labels):
        raise Exception('Some of the class names can not be found in the label file.')
    
    for label in target_labels:
        images_moved = []
        os.makedirs(f'{output_path}/{dataset}/{label}', exist_ok=True)
        
        for filename in labels[label]:
            old_path = f'{path}/{filename}.jpg'
            new_path = f'{output_path}/{dataset}/{filename}.jpg'
            
            shutil.copy(old_path, new_path)
            images_moved.append(new_path)
        print(f'Copied {len(images_moved)} images from {dataset} dataset {label} class...')
            
            

def get_labels(path, meta_dir='meta', filename='file.json'):
    '''
    Get list of labels from a file.
    '''
    if os.path.exists(f'{path}/{meta_dir}'):
        meta_path = f'{path}/{meta_dir}'
        with open(f'{meta_path}/{filename}') as f:
            return json.load(f)
    else:
        meta_path = ''
        for new_path in path.split('/'):
            meta_path += new_path + '/'
            if (os.path.exists(f'{meta_path}{meta_dir}')):
                meta_path += f'{meta_dir}/'
                with open(meta_path + filename) as f:
                    return json.load(f)
        raise FileNotFoundError(f'There is no {meta_dir}/{filename} inside {path}')