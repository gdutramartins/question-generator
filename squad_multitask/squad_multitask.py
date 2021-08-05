# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
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

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import nltk
nltk.download('punkt')

import nlp


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Stanford Question Answering Dataset (SQuAD) is a reading comprehension \
dataset, consisting of questions posed by crowdworkers on a set of Wikipedia \
articles, where the answer to every question is a segment of text, or span, \
from the corresponding reading passage, or the question might be unanswerable.
"""

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]


class SquadMultitaskConfig(nlp.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(SquadMultitaskConfig, self).__init__(**kwargs)


class SquadMultitask(nlp.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    _URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    _DEV_FILE = "dev-v1.1.json"
    _TRAINING_FILE = "train-v1.1.json"

    BUILDER_CONFIGS = [
        SquadMultitaskConfig(
            name=f"highlight_qg_format",
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
            description="Plain text"
        )        
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "source_text": nlp.Value("string"),
                    "target_text": nlp.Value("string"),
                    "task": nlp.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self._URL, self._TRAINING_FILE),
            "dev": os.path.join(self._URL, self._DEV_FILE),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]
    
    def process_e2e_qg(self, paragraph):
        source_text = f"generate questions: {paragraph['context'].strip()}"
        questions = [qas['question'].strip() for qas in paragraph['qas']]
        target_text = " {sep_token} ".join(questions)
        target_text = f"{target_text} {{sep_token}}"
        return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

    
    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        count = 0
        with open(filepath) as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    
                    yield count, self.process_e2e_qg(paragraph)
                    count += 1
