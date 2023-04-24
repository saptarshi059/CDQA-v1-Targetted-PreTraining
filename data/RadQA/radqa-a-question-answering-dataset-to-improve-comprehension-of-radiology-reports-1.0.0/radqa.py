# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""RadQA_for_HF: Connor Heaton/Saptarshi Sengupta"""

from datasets.tasks import QuestionAnsweringExtractive
import datasets
import requests
import json
import os

logger = datasets.logging.get_logger(__name__)

# You can copy an official description
_DESCRIPTION = """\
RadQA for loading from HuggingFace hub.
"""

_LICENSE = "Apache License 2.0"


class RadQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="RadQA", version=VERSION, description="RadQA for loading from HuggingFace hub."),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "document_id": datasets.Value("int32"),
                "title": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "is_impossible": datasets.Value("bool"),
                "id": datasets.Value("int32"),
                "answers": datasets.features.Sequence(
                    {
                        "text": datasets.Value("string"),
                        "answer_start": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            license=_LICENSE,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):

        # This code will be removed once the directory becomes public
        auth = ('saptarshi059', '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')

        train_url = 'https://raw.githubusercontent.com/saptarshi059/CDQA-v1-whole-entity-approach/main/data/RadQA' \
                    '/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/train.json'

        dev_url = 'https://raw.githubusercontent.com/saptarshi059/CDQA-v1-whole-entity-approach/main/data/RadQA/radqa' \
                  '-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/dev.json'

        test_url = 'https://raw.githubusercontent.com/saptarshi059/CDQA-v1-whole-entity-approach/main/data/RadQA' \
                   '/radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0/test.json' \

        os.mkdir('radqa_downloaded')

        train_request = requests.get(train_url, auth=auth)
        with open('radqa_downloaded/train.json', 'w') as f:
            json.dump(train_request.json(), f)

        dev_request = requests.get(dev_url, auth=auth)
        with open('radqa_downloaded/dev.json', 'w') as f:
            json.dump(dev_request.json(), f)

        test_request = requests.get(test_url, auth=auth)
        with open('radqa_downloaded/test.json', 'w') as f:
            json.dump(test_request.json(), f)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": 'radqa_downloaded/train.json'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": 'radqa_downloaded'
                                                                                            '/dev.json'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": 'radqa_downloaded/test.json'}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            radqa = json.load(f)
            for article in radqa["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        is_impossible = qa["is_impossible"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "is_impossible": is_impossible,
                            "id": qa["id"],
                            "document_id": paragraph["document_id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1
