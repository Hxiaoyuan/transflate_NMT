# -*- coding: utf-8 -*-
from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
import torch
from onmt.utils.logging import init_logger, logger
import onmt.inputters as inputters
import config.config as config
from collections import defaultdict, Counter
from functools import partial
from itertools import islice, repeat
from multiprocessing import Pool
import gc
from onmt.inputters.inputter import _build_fields_vocab, old_style_vocab, load_old_vocab


def _split_corpus(path, shard_size):
    """Yield a `list` containing `shard_size` line of `path`.
    """
    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard


def split_corpus(path, shard_size, default=None):
    if path is not None:
        return _split_corpus(path, shard_size)
    else:
        return repeat(default)


def maybe_load_vocab(corpus_type, counters, opt):
    src_vocab = None
    tgt_vocab = None
    existing_fields = None
    if corpus_type == config.train:
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                # src_vocab, src_vocab_size = _load_vocab(
                #     opt.src_vocab, "src", counters,
                #     opt.src_words_min_frequency)
        if opt.tgt_vocab != "":
            logger.error("opt.tgt_vocab 不为空")
            # tgt_vocab, tgt_vocab_size = _load_vocab(
            #     opt.tgt_vocab, "tgt", counters,
            #     opt.tgt_words_min_frequency)
    return src_vocab, tgt_vocab, existing_fields


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader,
                       align_reader, opt):
    assert corpus_type in [config.train, config.valid]

    if corpus_type == config.train:
        counters = defaultdict(Counter)
        srcs = opt.train_src
        tgts = opt.train_tgt
        ids = opt.train_ids
        aligns = opt.train_align
    elif corpus_type == config.valid:
        counters = None
        srcs = [opt.valid_src]
        tgts = [opt.valid_tgt]
        ids = [None]
        aligns = [opt.valid_align]

    src_vocab, tgt_vocab, existing_fields = maybe_load_vocab(
        corpus_type, counters, opt
    )
    existing_shards = []

    def shard_iterator(srcs, tgts, ids, aligns, existing_shards,
                       existing_fields, corpus_type, opt):
        for src, tgt, maybe_id, maybe_align in zip(srcs, tgts, ids, aligns):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}"
                                   .format(maybe_id))
                else:
                    if corpus_type == config.train:
                        assert existing_fields is not None, \
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if ((corpus_type == "train" or opt.filter_valid)
                    and tgt is not None):
                filter_pred = partial(
                    inputters.filter_example,
                    use_src_len=opt.data_type == "text",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            tgt_shards = split_corpus(tgt, opt.shard_size)
            align_shards = split_corpus(maybe_align, opt.shard_size)
            for i, (ss, ts, a_s) in enumerate(
                    zip(src_shards, tgt_shards, align_shards)):
                yield (i, (ss, ts, a_s, maybe_id, filter_pred))

    shard_iter = shard_iterator(srcs, tgts, ids, aligns, existing_shards,
                                existing_fields, corpus_type, opt)
    with Pool(opt.num_threads) as p:
        dataset_params = (
            corpus_type, fields, src_reader, tgt_reader,
            align_reader, opt, existing_fields,
            src_vocab, tgt_vocab
        )
        func = partial(process_one_shard, dataset_params)
        for sub_counter in p.imap(func, shard_iter):
            if sub_counter is not None:
                for key, value in sub_counter.items():
                    counters[key].update(value)
    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        new_fields = _build_fields_vocab(
            fields, counters, opt.data_type,
            opt.share_vocab, opt.vocab_size_multiple,
            opt.src_vocab_size, opt.src_words_min_frequency,
            opt.tgt_vocab_size, opt.tgt_words_min_frequency,
            subword_prefix=opt.subword_prefix,
            subword_prefix_is_joiner=opt.subword_prefix_is_joiner)
        if existing_fields is None:
            fields = new_fields
        else:
            fields = existing_fields

        if old_style_vocab(fields):
            fields = load_old_vocab(
                fields, opt.data_type, dynamic_dict=opt.dynamic_dict)

        # patch corpus_id
        if fields.get("corpus_id", False):
            fields["corpus_id"].vocab = new_fields["corpus_id"].vocab_cls(
                counters["corpus_id"])

        torch.save(fields, vocab_path)


def process_one_shard(corpus_params, params):
    corpus_type, fields, src_reader, tgt_reader, align_reader, opt, \
    existing_fields, src_vocab, tgt_vocab = corpus_params
    i, (src_shard, tgt_shard, align_shard, maybe_id, filter_pred) = params
    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    assert len(src_shard) == len(tgt_shard)
    logger.info("Building shard %d." % i)

    src_data = {"reader": src_reader, "data": src_shard, "dir": opt.src_dir}
    tgt_data = {"reader": tgt_reader, "data": tgt_shard, "dir": None}
    align_data = {"reader": align_reader, "data": align_shard, "dir": None}
    _readers, _data, _dir = inputters.Dataset.config(
        [('src', src_data), ('tgt', tgt_data), ('align', align_data)])

    dataset = inputters.Dataset(
        fields, readers=_readers, data=_data, dirs=_dir,
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred,
        corpus_id=maybe_id
    )
    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            sub_sub_counter['corpus_id'].update(
                ["train" if maybe_id is None else maybe_id])
            for name, field in fields.items():
                if (opt.data_type == "audio") and (name == "src"):
                    continue
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and
                                 src_vocab is not None) or \
                                (sub_n == 'tgt' and
                                 tgt_vocab is not None)
                    if (hasattr(sub_f, 'sequential')
                            and sub_f.sequential and not has_vocab):
                        val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt". \
        format(opt.save_data, shard_base, i)

    logger.info(" * saving %sth %s data shard to %s."
                % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def parser_init(parser):
    # parser.add('-config', '--config', required=False,
    #            is_config_file_arg=True, help='config file path')
    # parser.add('-save_config', '--save_config', required=False,
    #            is_write_out_config_file_arg=True,
    #            help='config file save path')
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text")

    group.add('--train_align', '-train_align', nargs='+', default=[None])
    group.add('--train_ids', '-train_ids', nargs='+', default=[None])
    group.add('--valid_align', '-valid_align', default=None)

    group.add('--src_dir', '-src_dir', default="")

    group.add('--max_shard_size', '-max_shard_size', type=int, default=0)

    group.add('--shard_size', '-shard_size', type=int, default=1000000,
              help="Divide src_corpus and tgt_corpus into "
                   "smaller multiple src_copus and tgt corpus files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")

    group.add('--overwrite', '-overwrite', action="store_true",
              help="Overwrite existing shards if any.")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    # if you want to pass an existing vocab.pt file, pass it to
    # -src_vocab alone as it already contains tgt vocab.
    group.add('--src_vocab', '-src_vocab', default="",
              help="Path to an existing source vocabulary. Format: "
                   "one word per line.")
    group.add('--tgt_vocab', '-tgt_vocab', default="",
              help="Path to an existing target vocabulary. Format: "
                   "one word per line.")
    group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
              type=str, default='',
              help="Path prefix to existing features vocabularies")
    group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")
    group.add('--vocab_size_multiple', '-vocab_size_multiple',
              type=int, default=1,
              help="Make the vocabulary size a multiple of this value")

    group.add('--src_words_min_frequency',
              '-src_words_min_frequency', type=int, default=0)
    group.add('--tgt_words_min_frequency',
              '-tgt_words_min_frequency', type=int, default=0)

    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
              help="Maximum source sequence length")
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
              help="Maximum target sequence length to keep.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")
    group.add('--lower', '-lower', action='store_true', help='lowercase data')
    group.add('--filter_valid', '-filter_valid', action='store_true',
              help='Filter validation data by src and/or tgt length')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add('--shuffle', '-shuffle', type=int, default=0,
              help="Shuffle data")
    group.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=100000,
              help="Report status every this many sentences")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    # group.add('--log_file_level', '-log_file_level', type=str,
    #           action=StoreLoggingLevelAction,
    #           choices=StoreLoggingLevelAction.CHOICES,
    #           default="0")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help="Window stride for spectrogram in seconds.")
    group.add('--window', '-window', default='hamming',
              help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3,
              choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")

    # Options for experimental source noising (BART style)
    group = parser.add_argument_group('Noise')
    group.add('--subword_prefix', '-subword_prefix',
              type=str, default="▁",
              help="subword prefix to build wordstart mask")
    group.add('--subword_prefix_is_joiner', '-subword_prefix_is_joiner',
              action='store_true',
              help="mask will need to be inverted if prefix is joiner")


def preprocess(opt):
    # 参数的验证
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0

    fields = inputters.get_fields(
        src_nfeats,
        tgt_nfeats,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)
    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)
    align_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, src_reader, tgt_reader, align_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset(
            'valid', fields, src_reader, tgt_reader, align_reader, opt)

def _get_parser():
    parser = ArgumentParser(description='preprocess.py')
    parser.add('--train_src', '-train_src', required=False, nargs='+',
               default=['F:/Project/Python/selfProject/translate_NMT/data/src-train.txt'],
               help="Path(s) to the training source data")

    parser.add('--train_tgt', '-train_tgt', required=False, nargs='+',
               default=['F:/Project/Python/selfProject/translate_NMT/data/tgt-train.txt'],
               help="Path(s) to the training target data")

    parser.add('--valid_src', '-valid_src',
               default='F:/Project/Python/selfProject/translate_NMT/data/src-val.txt',
               help="Path to the validation source data")

    parser.add('--valid_tgt', '-valid_tgt',
               default='F:/Project/Python/selfProject/translate_NMT/data/tgt-val.txt',
               help="Path to the validation target data")

    parser.add('--save_data', '-save_data', required=False,
               default='F:/Project/Python/selfProject/translate_NMT/data/demo',
               help="Output file for the prepared data")

    parser.add('--num_threads', '-num_threads', type=int, default=1,
               help="Number of shards to build in parallel.")
    parser_init(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()

    preprocess(opt)


if __name__ == "__main__":
    main()
