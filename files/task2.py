import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer,AutoModelForQuestionAnswering, TrainingArguments, Trainer,AutoConfig,AutoModel
from transformers import DefaultDataCollator
from transformers import TrainingArguments
from transformers import HfArgumentParser
from transformers import Trainer
from datasets import load_dataset
import torch
from transformers import DistilBertModel
from datasets import load_dataset
from transformers import PreTrainedModel,PretrainedConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import numpy as np
import re
import string
import collections
from datasets import DatasetDict
import ast
from transformers import EarlyStoppingCallback

# squad = load_dataset("squad", split="train[:5000]")
# squad = squad.train_test_split(test_size=0.2)
# my_dataset = squad


my_dataset = load_dataset("csv", data_files="/home/vp.shivasan/interiit/data/Task2dataSet_train.csv",split="train[:]")
my_dataset = my_dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
config = AutoConfig.from_pretrained("microsoft/MiniLM-L12-H384-uncased")#AutoConfig.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = ast.literal_eval(answers[i])
        start_char = int(answer["answer_start"][0])
        end_char = int(answer["answer_start"][0]) + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["context"] = examples["context"]
    inputs["answer"] = answers
    return inputs

tokenized_data = my_dataset.map(preprocess_function, batched=True, remove_columns=my_dataset["train"].column_names)


class MiniLMQA(PreTrainedModel):
    def __init__(self,config: PretrainedConfig):
        # super(DistillBERTQA, config).__init__()
        super().__init__(config)
        self.MiniLM = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased") #DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.qa_outputs = torch.nn.Linear(384, 2)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, input_ids=None, attention_mask=None,start_positions=None,end_positions=None,return_dict=None):
        minilm_output = self.MiniLM(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = minilm_output[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + minilm_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=minilm_output.hidden_states,
            attentions=minilm_output.attentions,
        )

model = MiniLMQA(config)
# model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

data_collator = DefaultDataCollator()

LOG_NAME = "task2_MiniLM-mod_50epochs_2e-5_FINAL_ES"

training_args = TrainingArguments(
    output_dir="/home/vp.shivasan/interiit/task2/training_dir/"+ LOG_NAME,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    metric_for_best_model = 'eval_loss',
    greater_is_better = False,
    load_best_model_at_end = True,
    save_strategy = 'epoch'

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
)
trainer.args.save_total_limit = 3

trainer.train()
trainer.save_model() 
trainer.save_state()



##  Prediction ##


# # Metric calculation taken from https://rajpurkar.github.io/SQuAD-explorer/
# def normalize_answer(s):
#   """Lower text and remove punctuation, articles and extra whitespace."""
#   def remove_articles(text):
#     regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
#     return re.sub(regex, ' ', text)
#   def white_space_fix(text):
#     return ' '.join(text.split())
#   def remove_punc(text):
#     exclude = set(string.punctuation)
#     return ''.join(ch for ch in text if ch not in exclude)
#   def lower(text):
#     return text.lower()
#   return white_space_fix(remove_articles(remove_punc(lower(s))))

# def get_tokens(s):
#   if not s: return []
#   return normalize_answer(s).split()

# def compute_f1(a_gold, a_pred):
#   gold_toks = get_tokens(a_gold)
#   pred_toks = get_tokens(a_pred)
#   common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#   num_same = sum(common.values())
#   if len(gold_toks) == 0 or len(pred_toks) == 0:
#     # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#     return int(gold_toks == pred_toks)
#   if num_same == 0:
#     return 0
#   precision = 1.0 * num_same / len(pred_toks)
#   recall = 1.0 * num_same / len(gold_toks)
#   f1 = (2 * precision * recall) / (precision + recall)
#   return f1

# test_dataset = load_dataset("csv", data_files="/home/vp.shivasan/interiit/data/Task2dataSet_test.csv",split="train[:]")
# tokenized_test_data = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# preds,labels,metrics = trainer.predict(tokenized_test_data)
# start_idxs = np.argmax(preds[0],axis=1)
# end_idxs = np.argmax(preds[1],axis=1)
# contexts = tokenized_test_data['context']
# answers = tokenized_test_data['answer']
# assert len(contexts) == len(start_idxs) == len(answers) 
# F1s = []
# for i,(sidx,eidx) in enumerate(zip(start_idxs,end_idxs)):
#     context_para = contexts[i]
#     pred_answer = context_para[sidx:eidx+1]
#     gold_answer = ast.literal_eval(answers[i])['text'][0]
#     f = compute_f1(gold_answer,pred_answer)
#     F1s.append(f)

# print("Average F1 score: ",np.mean(F1s))
