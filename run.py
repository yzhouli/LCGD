import argparse
import logging
import os
import sys
import importlib
import torch

from utils import Utils
from helpers.BaseLoader import BaseLoader, DataLoader
from helpers.BaseRunner import BaseRunner
from utils.Utils import info_model


def parse_global_args(parser):
    parser.add_argument('-data_name', default='douban')
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-train_rate', type=float, default=0.8)
    parser.add_argument('-valid_rate', type=float, default=0.1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('-filter_num', type=int, default=5)
    parser.add_argument('-load_vocab', type=int, default=0,
                        help='1=load existing u2idx/idx2u instead of rebuilding from cascades.txt. '
                             'Required for misdirdect_plus, whose "news_<id> user ts" first chunk '
                             'is mis-parsed by build_index (would overwrite the canonical vocab).')
    parser.add_argument('--seed', type=int, default=1506)
    return parser


def main(model_class):
    logging.info('-' * 45 + 'BEGIN: ' + Utils.get_time() + ' ' + '-' * 45)
    Utils.init_seed(seed=args.seed)

    logging.info(Utils.format_arg_str(args, exclude_lst=[]))

    # match CUDA device ordering to nvidia-smi (PCI bus) so -gpu N selects the
    # physical GPU N shown by nvidia-smi, not CUDA's default fastest-first index
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        # the selected physical GPU is the only visible device -> always cuda:0,
        # so -gpu 1/2 work for parallel runs across the 3 GPUs
        args.device = torch.device('cuda:0')
    logging.info('Device: {}'.format(args.device))

    data_loader = BaseLoader(args)
    user_size, total_cascades, timestamps, train, valid, test = data_loader.split_data(
        args.train_rate,
        args.valid_rate,
        load_dict=bool(args.load_vocab))

    train_data = DataLoader(train, batch_size=args.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=args.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=args.batch_size, load_dict=True, cuda=False)

    model = model_class(args, data_loader)
    info_model(model)

    runner = BaseRunner(args)
    runner.run(model, train_data, valid_data, test_data, args)


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='LCGD', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()

    try:
        model_module = importlib.import_module(f'models.{init_args.model_name}')
        model_class = getattr(model_module, init_args.model_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module models/{init_args.model_name} not found.")
    except AttributeError:
        raise ValueError(f"Model class {init_args.model_name} not found in module models/{init_args.model_name}.")

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = model_class.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    log_args = [init_args.model_name, args.data_name]
    log_file_name = '__'.join(log_args).replace(' ', '__')
    # Optional per-worker suffix so several sweep workers on different GPUs do not
    # overwrite each other's live log / checkpoint (each writes a distinct path).
    _worker = os.environ.get('IDP_WORKER', '')
    if _worker:
        log_file_name = log_file_name + '__' + _worker
    args.log_file = 'log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    args.model_path = 'saved/{}/{}.pt'.format(init_args.model_name, log_file_name)

    Utils.check_dir(args.log_file)
    Utils.check_dir(args.model_path)
    logging.basicConfig(filename=args.log_file, level='INFO')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main(model_class)
