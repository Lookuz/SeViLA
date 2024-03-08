"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from lavis.common.const import SEVILA_ANSWERER_PROMPT_POSTFIX, SEVILA_LOCALIZER_PROMPT_POSTFIX

class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

class NExTQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.answerer_postfix = SEVILA_ANSWERER_PROMPT_POSTFIX
        self.localizer_postfix = SEVILA_LOCALIZER_PROMPT_POSTFIX

        self._build_class_labels()

    def _load_auxiliary_mappings(self):
        pass
    
    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)
    
    def _build_class_labels(self):
        self.class_labels = {
            "A" : 0, "B" : 1, "C" : 2, "D" : 3, "E" : 4
        }
        self.labels_to_options = {v : k for k, v in self.class_labels.items()}
        self.num_options = 5

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        vname, question, label = ann["video"], ann["question"], ann["answer"]
        vpath = os.path.join(self.vis_root, f"{vname}.mp4")
        
        # Extract visual features
        frms, indices, fps = self.vis_processor(vpath)
        frms = frms.permute(1, 0, 2, 3)
        assert len(frms) == self.vis_processor.n_frms

        # Extract answerer and localizer prompts
        qa_prompt = " ".join((
            f"Question: {question}", 
            *["Option {}: {}".format(self.labels_to_options[i], ann[f"a{i}"]) for i in range(self.num_options)],
            self.answerer_postfix
        ))

        loc_prompt = " ".join((
            f"Question: {question}",
            "Options: ({})".format(
                " ".join([ann[f"a{i}"] for i in range(self.num_options)])
            ),
            self.localizer_postfix
        ))

        answer = f"Option {self.labels_to_options[label]}"

        return {
            "video": frms,
            "text_input": qa_prompt,
            "localizer_input": loc_prompt,
            "answer": answer,
            "question_id": str(ann['qid']),
        }
