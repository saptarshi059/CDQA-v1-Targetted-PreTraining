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
Cleaned version of COVID-QA containing fixes as mentioned in <paper yet to be published>.
"""


_LICENSE = "Apache License 2.0"


_URL = "https://github.com/saptarshi059/CDQA-v2-Auxilliary-Loss/blob/main/data/covid_qa_cleaned_CS/"
_URLs = {"covid_qa_cleaned_CS": _URL + "covid_qa_cleaned_CS.json"}


class CovidQADeepsetCleaned(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="covid_qa_cleaned_CS", version=VERSION, description="Cleaned version of COVID-QA (deepset) by Connor Heaton & Saptarshi Sengupta"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "document_id": datasets.Value("int32"),
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
        
        #This code will be removed once the directory becomes public

        url = 'https://raw.githubusercontent.com/saptarshi059/CDQA-v2-Auxilliary-Loss/main/data/covid_qa_cleaned_CS/covid_qa_cleaned_CS.json'
        auth = ('saptarshi059', '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')

        r = requests.get(url, auth=auth)

        os.mkdir('my_temp')
        
        with open('my_temp/covid_qa_cleaned_CS.json', 'w') as f:
            json.dump(r.json(), f)


        #url = _URLs[self.config.name]
        #downloaded_filepath = dl_manager.download_and_extract(r)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": 'my_temp/covid_qa_cleaned_CS.json'},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            covid_qa = json.load(f)
            for article in covid_qa["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    document_id = paragraph["document_id"]
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        is_impossible = qa["is_impossible"]
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "document_id": document_id,
                            "context": context,
                            "question": question,
                            "is_impossible": is_impossible,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
