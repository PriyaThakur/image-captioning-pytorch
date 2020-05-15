import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_val_loader(transform,
               mode='val',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['val'], "mode must be one of 'val'."
    assert batch_size==1, "Please change batch_size to 1 if testing your model."
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    assert vocab_from_file==True, "Change vocab_from_file to True."
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        test_info = json.loads(open(annotations_file).read())
        
        self.ids, self.paths, self.cap_dict = [], [], {}
        
        for item in test_info['images']:
            self.ids.append(item['id'])
            self.cap_dict[item['id']] = {'file_name':item['file_name'], 'captions':[]}
            self.paths.append(item['file_name'])
        
        for item in test_info['annotations']:
            #if item['image_id'] in self.dict:
            tokens = nltk.tokenize.word_tokenize(str(item['caption']).lower())
            caption = []
            caption.extend([self.vocab(token) for token in tokens])
            caption = torch.Tensor(caption).long()
            self.cap_dict[item['image_id']]['captions'].append(caption)
        
    def __getitem__(self, index):
        ret_dict = {}
        ann_id = self.ids[index]
        path = self.cap_dict[ann_id]['file_name']

        # Convert image to tensor and pre-process using transform
        PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(PIL_image)

        ret_dict = {'image':image, 'captions':self.cap_dict[ann_id]['captions'], 'image_id':ann_id}

        return ret_dict

    def __len__(self):
        return len(self.ids)
