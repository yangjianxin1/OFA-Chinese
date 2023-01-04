from component.ofa.modeling_ofa import OFAModelForCaption
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
    BertTokenizerFast
)
from loguru import logger
from component.dataset import CaptionDataset
from component.argument import CaptionArguments
import argparse
import os
import json
from os.path import join
from component.datacollator import CaptionCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/train_ofa.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取参数配置
    parser = HfArgumentParser((CaptionArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # 记录训练参数
    with open(train_args_file, 'r', encoding='utf8') as f:
        train_args = json.load(f)
    with open(join(training_args.output_dir, 'train_args.json'), 'w', encoding='utf8') as f:
        json.dump(train_args, f, indent=2)
    # 设置随机种子
    set_seed(training_args.seed)
    # 初始化模型
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    model = OFAModelForCaption.from_pretrained(args.model_name_or_path)
    # 是否将encoder的权重冻结，仅对decoder进行finetune
    if args.freeze_encoder:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
    # 加载数据集
    train_dataset = CaptionDataset(args.train_caption_file, args.train_image_file)
    # 初始化collator
    data_collator = CaptionCollator(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 开始训练
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(join(training_args.output_dir, 'checkpoint-final'))

    # 评测验证集的指标
    if args.test_caption_file is not None and args.test_image_file is not None:
        logger.info("*** start test ***")
        test_dataset = CaptionDataset(args.test_caption_file, args.test_image_file)
        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == '__main__':
    main()
