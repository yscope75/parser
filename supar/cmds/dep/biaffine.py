# -*- coding: utf-8 -*-

import argparse

from supar import BiaffineDependencyParser, BiaffineDepParserWRelations
from supar.cmds.run import init
import torch


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=BiaffineDepParserWRelations)
    #parser.set_defaults(Parser=BiaffineDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'elmo', 'bert'], nargs='*', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'transformer', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--embed', default='glove-6b-100', help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    subparser.add_argument('--warmup_steps', type=int, default=0, help='number of warmup steps')
    subparser.add_argument('--n_bert_layers', type=int, default=1, help='number of layer to extract embedding')
    subparser.add_argument('--model_type', default='baseline', help='Type of model to run: baseline or custom')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    torch.manual_seed(1)
    main()
