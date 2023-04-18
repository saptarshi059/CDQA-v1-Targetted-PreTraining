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
"""covid_qa_cleaned_CS: Connor Heaton/Saptarshi Sengupta"""


from datasets.tasks import QuestionAnsweringExtractive
import datasets
import requests
import json
import os

logger = datasets.logging.get_logger(__name__)


# You can copy an official description
_DESCRIPTION = """\
PolicyQA for Huggingface datasets...
"""


_LICENSE = "Apache License 2.0"


_URL = "https://github.com/saptarshi059/CDQA-v1-whole-entity-approach/tree/main/data/policy-qa"
_URLs = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json"
}


class PolicyQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="policy-qa", version=VERSION, description="PolicyQA for Huggingface datasets..."),
    ]

    @property
    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answers": datasets.features.Sequence(
                    {
                        "text": datasets.Value("string"),
                        "answer_start": datasets.Value("int64"),
                    }
                )
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
        
        #This code will be removed once the directory becomes public

        train_url = 'https://github.com/saptarshi059/CDQA-v1-whole-entity-approach/tree/main/data/policy-qa/train.json'
        dev_url = 'https://github.com/saptarshi059/CDQA-v1-whole-entity-approach/tree/main/data/policy-qa/dev.json'
        test_url = 'https://github.com/saptarshi059/CDQA-v1-whole-entity-approach/tree/main/data/policy-qa/test.json'

        auth = ('saptarshi059', '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')
        os.mkdir('my_temp')

        train_request = requests.get(train_url, auth=auth)
        with open('my_temp/train.json', 'w') as f:
            json.dump(train_request.json(), f)

        with open('my_temp/dev.json', 'w') as f:
            json.dump(r.json(), f)

        with open('my_temp/test.json', 'w') as f:
            json.dump(r.json(), f)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": 'my_temp/train.json'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": 'my_temp/dev.json'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": 'my_temp/test.json'},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            policy_qa = json.load(f)
            for article in policy_qa['train']["data"][0]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            }
                        }
                        key += 1