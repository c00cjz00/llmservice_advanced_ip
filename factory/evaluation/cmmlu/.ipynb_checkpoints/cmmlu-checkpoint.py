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
import os

import datasets
import pandas as pd


_CITATION = """\
@article{li2023cmmlu,
  title={CMMLU: Measuring massive multitask language understanding in Chinese},
  author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
  journal={arXiv preprint arXiv:2306.09212},
  year={2023}
}
"""

_DESCRIPTION = """\
CMMLU is a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge and reasoning abilities of LLMs within the Chinese language and cultural context.
"""

_HOMEPAGE = "https://github.com/haonan-li/CMMLU"

_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"

_URL = "cmmlu.zip"

task_list = [
  "dentistry",
  "traditional_chinese_medicine_clinical_medicine",
  "clinical_psychology",
  "technical",
  "culinary_skills",
  "mechanical",
  "logic_reasoning",
  "real_estate",
  "general_principles_of_law",
  "finance_banking",
  "anti_money_laundering",
  "ttqav2",
  "marketing_management",
  "business_management",
  "organic_chemistry",
  "advance_chemistry",
  "physics",
  "secondary_physics",
  "human_behavior",
  "national_protection",
  "jce_humanities",
  "linear_algebra",
  "politic_science",
  "agriculture",
  "official_document_management",
  "financial_analysis",
  "pharmacy",
  "educational_psychology",
  "statistics_and_machine_learning",
  "management_accounting",
  "introduction_to_law",
  "computer_science",
  "veterinary_pathology",
  "accounting",
  "fire_science",
  "optometry",
  "insurance_studies",
  "pharmacology",
  "taxation",
  "education_(profession_level)",
  "economics",
  "veterinary_pharmacology",
  "nautical_science",
  "occupational_therapy_for_psychological_disorders",
  "trust_practice",
  "geography_of_taiwan",
  "physical_education",
  "auditing",
  "administrative_law",
  "basic_medical_science",
  "macroeconomics",
  "trade",
  "chinese_language_and_literature",
  "tve_design",
  "junior_science_exam",
  "junior_math_exam",
  "junior_chinese_exam",
  "junior_social_studies",
  "tve_mathematics",
  "tve_chinese_language",
  "tve_natural_sciences",
  "junior_chemistry",
  "music",
  "education",
  "three_principles_of_people",
  "taiwanese_hokkien",
  "engineering_math"
]


class CMMLUConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.1"), **kwargs)


class CMMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CMMLUConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        #data_dir = dl_manager.download_and_extract(_URL)
        data_dir="/workspace/app/evaluation/cmmlu/cmmlu"
        task_name = self.config.name
        print(os.path.join(data_dir, f"test/{task_name}.csv"))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"test/{task_name}.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"dev/{task_name}.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        #df = pd.read_csv(filepath, header=0, index_col=0, encoding="utf-8")
        df = pd.read_csv(filepath, header=0, encoding="utf-8")
        for i, instance in enumerate(df.to_dict(orient="records")):
            question = instance.pop("question", "")
            answer = instance.pop("answer", "")
            instance["question"] = question
            instance["answer"] = answer
            yield i, instance

