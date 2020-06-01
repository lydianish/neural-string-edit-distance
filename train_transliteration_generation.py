#!/usr/bin/env python3

"""Sequence generation using neural string edit distance.

The data directory is expected to contains files {train,eval,text}.txt with
tab-separated source and target strings.
"""

import argparse
import logging
import os

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.functional import F
from torchtext import data

from experiment import experiment_logging, get_timestamp, save_vocab
from models import (
    EditDistNeuralModelConcurrent, EditDistNeuralModelProgressive)
from transliteration_utils import (
    load_transliteration_data, decode_ids, char_error_rate)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("data_prefix", type=str)
    parser.add_argument("--em-loss", default=None, type=float)
    parser.add_argument("--sampled-em-loss", default=None, type=float)
    parser.add_argument("--nll-loss", default=None, type=float)
    parser.add_argument("--distortion-loss", default=None, type=float)
    parser.add_argument("--final-state-loss", default=None, type=float)
    parser.add_argument("--model-type", default='transformer',
                        choices=["transformer", "rnn", "embeddings", "cnn"])
    parser.add_argument("--embedding-dim", default=256, type=int)
    parser.add_argument("--window", default=3, type=int)
    parser.add_argument("--hidden-size", default=256, type=int)
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument("--no-enc-dec-att", default=False, action="store_true")
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for test data decoding.")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--delay-update", default=4, type=int,
                        help="Update model every N steps.")
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--src-tokenized", default=False, action="store_true",
                        help="If true, source side are space separated tokens.")
    parser.add_argument("--tgt-tokenized", default=False, action="store_true",
                        help="If true, target side are space separated tokens.")
    parser.add_argument("--patience", default=2, type=int,
                        help="Number of validations witout improvement before decreasing learning rate.")
    parser.add_argument("--lr-decrease-count", default=10, type=int,
                        help="Number learning rate decays before early stopping.")
    parser.add_argument("--lr-decrease-ratio", default=0.7, type=float,
                        help="Factor by which the learning rate is decayed.")
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Initial learning rate.")
    args = parser.parse_args()

    if args.nll_loss is None and args.em_loss is None and args.sampled_em_loss is None:
        parser.error("No loss was specified.")

    experiment_params = (
        args.data_prefix.replace("/", "_") +
        f"_model{args.model_type}" +
        f"_hidden{args.hidden_size}" +
        f"_attheads{args.attention_heads}" +
        f"_layers{args.layers}" +
        f"_encdecatt{not args.no_enc_dec_att}" +
        f"_window{args.window}" +
        f"_batch{args.batch_size}" +
        f"_dealy{args.delay_update}" +
        f"_patence{args.patience}" +
        f"_nll{args.nll_loss}" +
        f"_EMloss{args.em_loss}" +
        f"_sampledEMloss{args.sampled_em_loss}" +
        f"_finalStateLoss{args.final_state_loss}" +
        f"_distortion{args.distortion_loss}")
    experiment_dir = experiment_logging(
        f"edit_gen_{experiment_params}_{get_timestamp()}", args)
    model_path = os.path.join(experiment_dir, "model.pt")
    tb_writer = SummaryWriter(experiment_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ar_text_field, en_text_field, train_iter, val_iter, test_iter = \
        load_transliteration_data(
            args.data_prefix, args.batch_size, device,
            src_tokenized=args.src_tokenized,
            tgt_tokenized=args.tgt_tokenized)

    save_vocab(
        ar_text_field.vocab.itos, os.path.join(experiment_dir, "src_vocab"))
    save_vocab(
        en_text_field.vocab.itos, os.path.join(experiment_dir, "tgt_vocab"))

    model = EditDistNeuralModelProgressive(
        ar_text_field.vocab, en_text_field.vocab, device, directed=True,
        model_type=args.model_type,
        hidden_dim=args.hidden_size,
        hidden_layers=args.layers,
        attention_heads=args.attention_heads,
        window=args.window,
        encoder_decoder_attention=not args.no_enc_dec_att).to(device)

    kl_div = nn.KLDivLoss(reduction='none').to(device)
    nll = nn.NLLLoss(reduction='none').to(device)
    xent = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate)

    step = 0
    best_wer = 1e9
    best_wer_step = 0
    best_cer = 1e9
    best_cer_step = 0
    stalled = 0
    learning_rate = args.learning_rate
    remaining_decrease = args.lr_decrease_count

    for _ in range(args.epochs):
        if remaining_decrease <= 0:
            break

        for i, train_ex in enumerate(train_iter):
            if stalled > args.patience:
                learning_rate *= args.lr_decrease_ratio
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                remaining_decrease -= 1
                stalled = 0
                logging.info(f"Decreasing learning rate to {learning_rate}.")
            if remaining_decrease <= 0:
                break
            step += 1

            (action_scores, expected_counts,
                logprob, next_symbol_score, distorted_probs) = model(
                train_ex.ar, train_ex.en)

            en_mask = (train_ex.en != model.en_pad).float()
            ar_mask = (train_ex.ar != model.ar_pad).float()
            table_mask = (ar_mask.unsqueeze(2) * en_mask.unsqueeze(1)).float()

            loss = torch.tensor(0.).to(device)
            kl_loss = 0
            if args.em_loss is not None:
                tgt_dim = action_scores.size(-1)
                kl_loss_raw = kl_div(
                    action_scores.reshape(-1, tgt_dim),
                    expected_counts.reshape(-1, tgt_dim)).sum(1)
                kl_loss = (
                    (kl_loss_raw * table_mask.reshape(-1)).sum() /
                    table_mask.sum())
                loss += args.em_loss * kl_loss

            sampled_em_loss = 0
            if args.sampled_em_loss is not None:
                tgt_dim = action_scores.size(-1)
                # TODO do real sampling instead of argmax
                sampled_actions = expected_counts.argmax(3)
                #sampled_actions = torch.multinomial(expected_counts[:, 1:, 1:].reshape(-1, tgt_dim), 1)
                sampled_em_loss_raw = xent(
                    action_scores[:, 1:, 1:].reshape(-1, tgt_dim),
                    sampled_actions[:, 1:, 1:].reshape(-1))
                sampled_em_loss = (
                    (sampled_em_loss_raw * table_mask[:, 1:, 1:].reshape(-1)).sum() /
                    table_mask.sum())
                loss += args.sampled_em_loss * sampled_em_loss

            nll_loss = 0
            if args.nll_loss is not None:
                tgt_dim = next_symbol_score.size(-1)
                nll_loss_raw = nll(
                    next_symbol_score.reshape(-1, tgt_dim),
                    train_ex.en[:, 1:].reshape(-1))
                nll_loss = (
                    (en_mask[:, 1:].reshape(-1) * nll_loss_raw).sum() /
                    en_mask[:, 1:].sum())
                loss += args.nll_loss * nll_loss

            distortion_loss = 0
            if args.distortion_loss is not None:
                distortion_loss = (table_mask * distorted_probs).sum() / table_mask.sum()
                loss += args.distortion_loss * distortion_loss

            final_state_loss = 0
            if args.final_state_loss is not None:
                final_state_loss = -logprob.mean()
                loss += args.final_state_loss * final_state_loss

            loss.backward()

            if step % args.delay_update == args.delay_update - 1:
                logging.info(f"step: {step}, train loss = {loss:.3g} "
                      f"(NLL {nll_loss:.3g}, "
                      f"distortion: {distortion_loss:.3g}, "
                      f"final state NLL: {final_state_loss:.3g}, "
                      f"EM: {kl_loss:.3g}, "
                      f"sampled EM: {sampled_em_loss:.3g})")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % (args.delay_update * 50) == args.delay_update * 50 - 1:
                tb_writer.add_scalar('train/loss', loss, step)
                tb_writer.add_scalar('train/nll', nll_loss, step)
                tb_writer.add_scalar('train/em_kl_div', kl_loss, step)
                tb_writer.add_scalar(
                    'train/sampled_em_nll', sampled_em_loss, step)
                model.eval()

                sources = []
                ground_truth = []
                hypotheses = []

                for j, val_ex in enumerate(val_iter):
                    with torch.no_grad():
                        #decoded_val = model.beam_search(val_ex.ar, beam_size=2)
                        decoded_val = model.decode(val_ex.ar)

                        for ar, en, hyp in zip(val_ex.ar, val_ex.en, decoded_val):
                            src_string = decode_ids(
                                ar, ar_text_field, args.src_tokenized)
                            tgt_string = decode_ids(
                                en, en_text_field, args.tgt_tokenized)
                            hypothesis = decode_ids(
                                hyp, en_text_field, args.tgt_tokenized)

                            sources.append(src_string)
                            ground_truth.append(tgt_string)
                            hypotheses.append(hypothesis)

                        if j == 0:
                            for src, hyp, tgt in zip(sources[:10], hypotheses, ground_truth):
                                logging.info("")
                                logging.info(f"'{src}' -> '{hyp}' ({tgt})")

                logging.info("")

                wer = 1 - sum(
                    float(gt == hyp) for gt, hyp
                    in zip(ground_truth, hypotheses)) / len(ground_truth)
                cer = char_error_rate(
                    hypotheses, ground_truth, args.tgt_tokenized)

                stalled += 1
                if wer < best_wer:
                    best_wer = wer
                    best_wer_step = step
                    stalled = 0
                if cer < best_cer:
                    best_cer = cer
                    best_cer_step = step
                    stalled = 0

                logging.info(
                    f"WER: {wer:.3g}   (best {best_wer:.3g}, step {best_wer_step})")
                logging.info(
                    f"CER: {cer:.3g}   (best {best_cer:.3g}, step {best_cer_step})")
                if stalled > 0:
                    logging.info(f"Stalled {stalled} times.")
                else:
                    torch.save(model, model_path)

                logging.info("")

                tb_writer.add_scalar('val/cer', cer, step)
                tb_writer.add_scalar('val/wer', wer, step)
                tb_writer.flush()
                model.train()

    logging.info("TRAINING FINISHED, evaluating on test data")
    logging.info("")
    model = torch.load(model_path)
    model.eval()

    sources = []
    ground_truth = []
    hypotheses = []

    for j, test_ex in enumerate(test_iter):
        with torch.no_grad():
            decoded_val = model.beam_search(test_ex.ar, beam_size=args.beam_size)

            for ar, en, hyp in zip(test_ex.ar, test_ex.en, decoded_val):
                src_string = decode_ids(ar, ar_text_field, args.src_tokenized)
                tgt_string = decode_ids(en, en_text_field, args.tgt_tokenized)
                hypothesis = decode_ids(hyp, en_text_field, args.tgt_tokenized)

                sources.append(src_string)
                ground_truth.append(tgt_string)
                hypotheses.append(hypothesis)

            if j == 0:
                for src, hyp, tgt in zip(sources[:10], hypotheses, ground_truth):
                    logging.info("")
                    logging.info(f"'{src}' -> '{hyp}' ({tgt})")

    logging.info("")

    wer = 1 - sum(
        float(gt == hyp) for gt, hyp
        in zip(ground_truth, hypotheses)) / len(ground_truth)
    cer = char_error_rate(hypotheses, ground_truth, args.tgt_tokenized)

    logging.info(f"WER: {wer:.3g}")
    logging.info(f"CER: {cer:.3g}")
    logging.info("")


if __name__ == "__main__":
    main()
