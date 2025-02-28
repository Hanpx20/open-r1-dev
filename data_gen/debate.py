# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.argument_graph import ArgumentGraph
from verl.utils.debate_prompts import *
import argparse


def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)



if __name__ == '__main__':
    '''
    pos is the statement for the persuader
    neg is the statement for the opponent (which has a HIGHER agreement rate by the judge initially)
    The debate graph represents the opinion of the judge, so its root statement is 'neg'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/mmlu')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = "debate"


    dataset = datasets.load_dataset('json', data_files = {"train": "cmv/final/train.json", "test": "cmv/final/test.json"})

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if args.template_type == 'base':
                initial_judger = ArgumentGraph(args.model_name, example['pos'], root_statement_tilde=example['neg'], use_api=True)
                if initial_judger.tree.get_node(initial_judger.tree.root).data["confidence"] > 0.5:
                    example['pos'], example['neg'] = example['neg'], example['pos']
                question = debater_sys_prompt.replace('<root_statement>', example['pos']).replace('<root_statement_tilde>', example['neg']).replace("<name>", "Alice")
            else:
                raise ValueError(f"Invalid template_type: {args.template_type}")
        
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "system",
                    "content": question,
                }],
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "question": question,
                },
                "reward_model": {
                    "style": "debate"
                },
            }
            return data

        return process_fn

    train_dataset = dataset['train'].map(function=make_map_fn('train'), with_indices=True)
    test_dataset = dataset['test'].map(function=make_map_fn('test'), with_indices=True)
    print(f"The size of training set: {len(train_dataset)}")
    print(f"The size of testing set: {len(test_dataset)}")
    
    local_dir = os.path.join(args.local_dir, args.model_name.split('/')[-1])
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    parquet_to_json(os.path.join(local_dir, 'train.parquet'), os.path.join(local_dir, 'train.jsonl'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    parquet_to_json(os.path.join(local_dir, 'test.parquet'), os.path.join(local_dir, 'test.jsonl'))
    
    
    
