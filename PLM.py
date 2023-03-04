import math
import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm
from transformers import (RobertaTokenizer, PLBartConfig, PLBartTokenizer, PLBartForConditionalGeneration,
                          RobertaConfig, RobertaModel, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          T5Config, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup)
import logging

from datasets import read_examples, convert_examples_to_features
# from utils import get_bleu_socre
from utils import compute

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def build_or_load_gen_model(model_type, model_name_or_path, load_model_path):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)
    # model = torch.compile(model)
    if load_model_path is not None:
        logger.info("Reload model from {}".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))
    return config, model, tokenizer

class Encoder_Decoder():
    def __init__(self, model_type, model_name_or_path, load_model_path, beam_size, max_source_length, max_target_length, method_type='enc-dec'):
        self.model_type = model_type
        self.config, self.model, self.tokenizer = build_or_load_gen_model(model_type, model_name_or_path,
                                                                          load_model_path)
        self.method_type = method_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.beam_size, self.max_source_length, self.max_target_length = beam_size, max_source_length, max_target_length


    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop, task,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu):
        
        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length,
                                                      self.max_target_length, stage='train')
        
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num/train_batch_size))
        logger.info("  Num epoch = %d", num_train_epochs)
        dev_dataset = {}
        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0
        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                     labels=target_ids, decoder_attention_mask=target_mask)
                total_loss = outputs.loss
                tr_loss += total_loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
        
            if do_eval==True:
                # Eval model with dev dataset
                eval_examples = read_examples(eval_filename)
                eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                             self.max_target_length, stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                logger.info("***** Running evaluation  *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", eval_batch_size)
                logger.info("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                                 labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss
                    eval_loss = eval_loss + loss.item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu:
                    self.model.eval()
                    df = pd.read_csv(eval_filename)
                    to_predict = df["src"].tolist()
                    ref_list = df["tgt"].tolist()
                    all_outputs = []
                    # Batching
                    for batch in tqdm(
                            [to_predict[i: i + eval_batch_size] for i in range(0, len(to_predict), eval_batch_size)],
                            desc="Generating outputs", ):
                        input = self.tokenizer.batch_encode_plus(
                            batch,
                            max_length=self.max_source_length,
                            padding="max_length",
                            return_tensors="pt",
                            truncation=True,
                        )
                        input_ids = input["input_ids"].to(self.device)
                        source_mask = input["attention_mask"].to(self.device)
                        outputs = self.model.generate(input_ids,
                                                      attention_mask=source_mask,
                                                      num_beams=self.beam_size,
                                                      max_length=self.max_target_length)
                        all_outputs.extend(outputs.cpu().numpy())
                    hyp_list = [
                        self.tokenizer.decode(
                            output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        for output_id in all_outputs
                    ]

                    assert len(ref_list) == len(hyp_list)
                    df = pd.DataFrame(hyp_list)
                    df.to_csv("hyp_temp.csv", index=False, header=False)
                    df = pd.DataFrame(ref_list)
                    df.to_csv("ref_temp.csv", index=False, header=False)

                    bleu4=compute("hyp_temp.csv", "ref_temp.csv")['Bleu_4']

                    logger.info('dev set: bleu = %.2f\n' % bleu4)
                    logger.info("  " + "*" * 20)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if bleu4 > best_bleu:
                        df = pd.DataFrame(hyp_list)
                        df.to_csv(output_dir+"preds.csv", index=False, header=False)
                        df = pd.DataFrame(ref_list)
                        df.to_csv(output_dir+"golds.csv", index=False, header=False)
                        count = 0
                        logger.info("  Best bleu:%s", bleu4)
                        logger.info("  " + "*" * 20)
                        best_bleu = bleu4
                        # Save best checkpoint for best bleu
                        output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir_bleu):
                            os.makedirs(output_dir_bleu)
                        model_to_save = self.model.module if hasattr(self.model,
                                                                'module') else self.model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        count += 1
                        if count == early_stop:
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    def test(self, batch_size, filename, output_dir):
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", batch_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df = pd.read_csv(filename)

        to_predict = df["src"].tolist()
        ref_list = df["tgt"].tolist()

        all_outputs = []
        # Batching
        for batch in tqdm(
                [to_predict[i: i + batch_size] for i in range(0, len(to_predict), batch_size)],
                desc="Generating outputs", ):
            input = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.max_source_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids = input["input_ids"].to(self.device)
            source_mask = input["attention_mask"].to(self.device)
            outputs = self.model.generate(input_ids,
                                              attention_mask=source_mask,
                                              num_beams=self.beam_size,
                                              do_sample=True,
                                              max_length=self.max_target_length)
            all_outputs.extend(outputs.cpu().numpy())

        hyp_list = [
            self.tokenizer.decode(
                output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for output_id in all_outputs
        ]

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir+"/gold.csv", index=False, header=False)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/pred.csv", index=False, header=False)
        bleu4 = compute("hyp_temp.csv", "ref_temp.csv")['Bleu_4']

        logger.info('dev set: bleu = %.2f\n' % bleu4)
        logger.info("  " + "*" * 20)
