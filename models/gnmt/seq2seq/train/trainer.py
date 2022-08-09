# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import time
from itertools import cycle

import numpy as np
import torch
import torch.optim
import torch.utils.data
#from apex.parallel import DistributedDataParallel
#from apex import amp
from torch.nn.parallel import DistributedDataParallel

from seq2seq.train.fp_optimizers import FP16Optimizer
from seq2seq.train.fp_optimizers import FP32Optimizer
from seq2seq.train.fp_optimizers import AMPOptimizer
from seq2seq.train.lr_scheduler import WarmupMultiStepLR
from seq2seq.utils import AverageMeter
from seq2seq.utils import sync_workers

import horovod.torch as hvd

class Seq2SeqTrainer:
    """
    Seq2SeqTrainer
    """
    def __init__(self,
                 model,
                 batch_first,
                 criterion,
                 opt_config,
                 scheduler_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 save_info={},
                 save_dir='.',
                 train_iterations=0,
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 loss_scaling={},
                 intra_epoch_eval=0,
                 prealloc_mode='always',
                 iter_size=1,
                 translator=None,
                 verbose=False,
                 num_minibatches=None,
                 profile_dir=None,
                 no_optimizer=False):
        """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param opt_config: dictionary with options for the optimizer
        :param scheduler_config: dictionary with options for the learning rate
            scheduler
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_freq: saves checkpoint every 'save_freq' iterations
        :param grad_clip: coefficient for gradient clipping
        :param save_info: dict with additional state stored in each checkpoint
        :param save_dir: path to the directiory for checkpoints
        :param train_iterations: total number of training iterations to execute
        :param checkpoint_filename: name of files with checkpoints
        :param keep_checkpoints: max number of checkpoints to keep
        :param math: arithmetic type
        :param loss_scaling: options for dynamic loss scaling
        :param intra_epoch_eval: number of additional eval runs within each
            training epoch
        :param prealloc_mode: controls preallocation,
            choices=['off', 'once', 'always']
        :param iter_size: number of iterations between weight updates
        :param translator: instance of Translator, runs inference on test set
        :param verbose: enables verbose logging
        """
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.device = next(model.parameters()).device
        self.print_freq = print_freq
        self.verbose = verbose
        self.loss = None
        self.translator = translator
        self.intra_epoch_eval = intra_epoch_eval
        self.iter_size = iter_size
        self.prealloc_mode = prealloc_mode
        self.preallocated = False

        #self.distributed = torch.distributed.is_initialized()
        self.batch_first = batch_first

        self.num_minibatches = num_minibatches
        self.profile_dir = profile_dir
        self.no_optimizer = no_optimizer

        params = self.model.parameters()

        if math == 'manual_fp16':
            self.fp_optimizer = FP16Optimizer(
                self.model, grad_clip,
                loss_scale=loss_scaling['init_scale'],
                dls_upscale_interval=loss_scaling['upscale_interval']
                )
            params = self.fp_optimizer.fp32_params
        elif math == 'fp32':
            self.fp_optimizer = FP32Optimizer(self.model, grad_clip)

        opt_name = opt_config.pop('optimizer')
        self.optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
        logging.info(f'Using optimizer: {self.optimizer}')

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

        self.optimizer = hvd.DistributedOptimizer(
                self.optimizer,
                named_parameters=self.model.named_parameters())

        self.scheduler = WarmupMultiStepLR(self.optimizer, train_iterations,
                                           **scheduler_config)

    def iterate(self, src, tgt, update=True, training=True, dummy_input=False):
        """
        Performs one iteration of the training/validation.

        :param src: batch of examples from the source language
        :param tgt: batch of examples from the target language
        :param update: if True: optimizer does update of the weights
        :param training: if True: executes optimizer
        """
        forward_time = 0
        backward_time = 0
        gradient_time = 0
        update_time = 0
        if dummy_input:
            LOCAL_BATCH_SIZE = self.batch_size
            print(f'dummy input!! batch size: {LOCAL_BATCH_SIZE}')
            if self.batch_first:
                src = torch.ones(LOCAL_BATCH_SIZE, 50, dtype=torch.int64)
                src_length = torch.ones(LOCAL_BATCH_SIZE, dtype=torch.int64)
                tgt = torch.ones(LOCAL_BATCH_SIZE, 50, dtype=torch.int64)
            else:
                src = torch.ones(50, LOCAL_BATCH_SIZE, dtype=torch.int64)
                src_length = torch.ones(LOCAL_BATCH_SIZE, dtype=torch.int64)
                tgt = torch.ones(50, LOCAL_BATCH_SIZE, dtype=torch.int64)
            # Filter data by min:0 and max:50
            tgt_length = torch.randint(1, 49, (LOCAL_BATCH_SIZE,))
        else:
            src, src_length = src
            tgt, tgt_length = tgt
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_length = src_length.to(self.device)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))
        if training:
            start = time.time()
        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)
        else:
            output = self.model(src, src_length, tgt[:-1])
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)
        loss = self.criterion(output.view(T * B, -1),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()
        loss /= (B * self.iter_size)
        if training:
            forward_time = time.time() - start
        if training:
            bwd_start = time.time()
            loss.backward()
            gradient_time = time.time() - bwd_start
            update_start = time.time()
            if not self.no_optimizer:
                self.fp_optimizer.step(loss, self.optimizer, self.scheduler,
                                    update)
            self.optimizer.step()
            update_time = time.time() - update_start
            backward_time = time.time() - bwd_start

        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / B
        return loss_per_token, loss_per_sentence, num_toks, \
                forward_time, backward_time, gradient_time, update_time

    def feed_data(self, data_loader, training=True):
        """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval+2)[1:-1]
            iters_with_update = len(data_loader) // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        forward_time = AverageMeter()
        gradient_time = AverageMeter()
        update_time = AverageMeter()
        backward_time = AverageMeter()
        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        batch_size = data_loader.batch_size
        self.batch_size = batch_size

        is_outlier = False
        end = time.time()
        for i, (src, tgt) in enumerate(data_loader):
            if self.num_minibatches is not None and i > self.num_minibatches:
                break

            self.save_counter += 1
            # measure data loading time
            if i >= 10:
                loading_time = time.time() - end
                if loading_time >= 1 and self.num_minibatches is not None:
                    print(f'[WARNING] loading outlier: {loading_time:6.5f}')
                    is_outlier = True
                if not is_outlier:
                    data_time.update(loading_time)

            update = False
            if i % self.iter_size == self.iter_size - 1:
                update = True

            # do a train/evaluate iteration
            stats = self.iterate(src, tgt, update, training=training)
            loss_per_token, loss_per_sentence, num_toks, \
                iter_fwd_time, iter_bwd_time , iter_grad_time, iter_update_time = stats

            # measure accuracy and record loss
            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            # measure elapsed time
            elapsed = time.time() - end
            if i >= 10 and not is_outlier:
                batch_time.update(elapsed)
                forward_time.update(iter_fwd_time)
                backward_time.update(iter_bwd_time)
                gradient_time.update(iter_grad_time)
                update_time.update(iter_update_time)
                src_tok_time.update(num_toks['src'] / elapsed)
                tgt_tok_time.update(num_toks['tgt'] / elapsed)
                tot_num_toks = num_toks['tgt'] + num_toks['src']
                tot_tok_time.update(tot_num_toks / elapsed)
            self.loss = losses_per_token.avg

            if training and i in eval_iters:
                eval_fname = f'eval_epoch_{self.epoch}_iter_{i}'
                eval_path = os.path.join(self.save_dir, eval_fname)
                _, eval_stats = self.translator.run(
                    calc_bleu=True,
                    epoch=self.epoch,
                    iteration=i,
                    eval_path=eval_path,
                    )
                test_bleu = eval_stats['bleu']

                log = []
                log += [f'TRAIN [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'BLEU: {test_bleu:.2f}']
                log = '\t'.join(log)
                logging.info(log)

                self.model.train()
                self.preallocate(data_loader.batch_size,
                                 data_loader.dataset.max_len, training=True)

            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'VALIDATION'
                log = []
                log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Data {data_time.val:.4f} ({data_time.avg:.4f})']
                log += [f'Forward {forward_time.val:.5f} ({forward_time.avg:.5f})']
                log += [f'Backward {backward_time.val:.5f} ({backward_time.avg:.5f})']
                log += [f'Gradient {gradient_time.val:.5f} ({gradient_time.avg:.5f})']
                log += [f'Update {update_time.val:.5f} ({update_time.avg:.5f})']
                log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                if self.verbose:
                    log += [f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})']
                    log += [f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})']
                    log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                if training:
                    lr = self.optimizer.param_groups[0]['lr']
                    log += [f'LR {lr:.3e}']
                log = '\t'.join(log)
                logging.info(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    with sync_workers() as rank:
                        if rank == 0:
                            self.save(identifier=identifier)

            end = time.time()

        return losses_per_token.avg, tot_tok_time.avg

    def profile_feed_data(self, data_loader, training=True):
        """
        Runs training or validation on batches from data_loader.

        :param data_loader: data loader
        :param training: if True runs training else runs validation
        """
        if training:
            assert self.optimizer is not None
            eval_fractions = np.linspace(0, 1, self.intra_epoch_eval+2)[1:-1]
            iters_with_update = len(data_loader) // self.iter_size
            eval_iters = (eval_fractions * iters_with_update).astype(int)
            eval_iters = eval_iters * self.iter_size
            eval_iters = set(eval_iters)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        batch_size = data_loader.batch_size
        self.batch_size = batch_size

        end = time.time()
        wait, warmup, active, repeat = 10, 1, 1, 1
        max_profile_step = (wait + warmup + active) * repeat
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
                record_shapes=True,
                with_stack=True
                ) as prof:
            # Remove data loading overhead
            #for i in range(max_profile_step + 1):
            for i, (src, tgt) in enumerate(data_loader):
                if i >= max_profile_step:
                    logging.info(f'[INFO] step: {i} - profiler finishes.')
                    break
                self.save_counter += 1
                # measure data loading time
                data_time.update(time.time() - end)

                update = False
                if i % self.iter_size == self.iter_size - 1:
                    update = True

                # do a train/evaluate iteration
                src, tgt = None, None
                stats = self.iterate(src, tgt, update, training=training, dummy_input=True)
                prof.step()

                loss_per_token, loss_per_sentence, num_toks, _, _, _, _ = stats

                # measure accuracy and record loss
                losses_per_token.update(loss_per_token, num_toks['tgt'])
                losses_per_sentence.update(loss_per_sentence, batch_size)

                # measure elapsed time
                elapsed = time.time() - end
                batch_time.update(elapsed)
                src_tok_time.update(num_toks['src'] / elapsed)
                tgt_tok_time.update(num_toks['tgt'] / elapsed)
                tot_num_toks = num_toks['tgt'] + num_toks['src']
                tot_tok_time.update(tot_num_toks / elapsed)
                self.loss = losses_per_token.avg

                if training and i in eval_iters:
                    eval_fname = f'eval_epoch_{self.epoch}_iter_{i}'
                    eval_path = os.path.join(self.save_dir, eval_fname)
                    _, eval_stats = self.translator.run(
                        calc_bleu=True,
                        epoch=self.epoch,
                        iteration=i,
                        eval_path=eval_path,
                        )
                    test_bleu = eval_stats['bleu']

                    log = []
                    log += [f'TRAIN [{self.epoch}][{i}/{len(data_loader)}]']
                    log += [f'BLEU: {test_bleu:.2f}']
                    log = '\t'.join(log)
                    logging.info(log)

                    self.model.train()
                    self.preallocate(data_loader.batch_size,
                                    data_loader.dataset.max_len, training=True)

                if i % self.print_freq == 0:
                    phase = 'TRAIN' if training else 'VALIDATION'
                    log = []
                    log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
                    log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                    log += [f'Data {data_time.val:.2e} ({data_time.avg:.2e})']
                    log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                    if self.verbose:
                        log += [f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})']
                        log += [f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})']
                        log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                    log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                    if training:
                        lr = self.optimizer.param_groups[0]['lr']
                        log += [f'LR {lr:.3e}']
                    log = '\t'.join(log)
                    logging.info(log)

                save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
                if training and save_chkpt:
                    self.save_counter = 0
                    self.save_info['iteration'] = i
                    identifier = next(self.checkpoint_counter, -1)
                    if identifier != -1:
                        with sync_workers() as rank:
                            if rank == 0:
                                self.save(identifier=identifier)

                end = time.time()

        tot_tok_time.reduce('sum')
        losses_per_token.reduce('mean')

        return losses_per_token.avg, tot_tok_time.avg

    def preallocate(self, batch_size, max_length, training):
        """
        Generates maximum sequence length batch and runs forward and backward
        pass without updating model parameters.

        :param batch_size: batch size for preallocation
        :param max_length: max sequence length for preallocation
        :param training: if True preallocates memory for backward pass
        """
        if self.prealloc_mode == 'always' or (self.prealloc_mode == 'once' and
                                              not self.preallocated):
            logging.info('Executing preallocation')
            torch.cuda.empty_cache()

            src_length = torch.full((batch_size,), max_length,
                                    dtype=torch.int64)
            tgt_length = torch.full((batch_size,), max_length,
                                    dtype=torch.int64)

            if self.batch_first:
                shape = (batch_size, max_length)
            else:
                shape = (max_length, batch_size)

            src = torch.full(shape, 4, dtype=torch.int64)
            tgt = torch.full(shape, 4, dtype=torch.int64)
            src = src, src_length
            tgt = tgt, tgt_length
            self.iterate(src, tgt, update=False, training=training)
            self.model.zero_grad()
            self.preallocated = True

    def optimize(self, data_loader):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(True)
        self.model.train()
        self.preallocate(data_loader.batch_size, data_loader.dataset.max_len,
                         training=True)

        if self.profile_dir:
            logging.info('[WARNING] Running with profiler degrades performance.')
            output = self.profile_feed_data(data_loader, training=True)
        else:
            output = self.feed_data(data_loader, training=True)

        self.model.zero_grad()
        return output

    def evaluate(self, data_loader):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        # To save the memory when evaluate
        #self.preallocate(data_loader.batch_size, data_loader.dataset.max_len,
        #                 training=False)

        output = self.feed_data(data_loader, training=False)

        self.model.zero_grad()
        return output

    def load(self, filename):
        """
        Loads checkpoint from filename.

        :param filename: path to the checkpoint file
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
            self.model.load_state_dict(checkpoint['state_dict'])
            self.fp_optimizer.initialize_model(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info(f'Loaded checkpoint {filename} (epoch {self.epoch})')
        else:
            logging.error(f'Invalid checkpoint: {filename}')

    def save(self, identifier=None, is_best=False, save_all=False):
        """
        Stores checkpoint to a file.

        :param identifier: identifier for periodic checkpoint
        :param is_best: if True stores checkpoint to 'model_best.pth'
        :param save_all: if True stores checkpoint after completed training
            epoch
        """

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_dir, filename)
            logging.info(f'Saving model to {filename}')
            torch.save(state, filename)

        model_state = self.model.state_dict()

        state = {
            'epoch': self.epoch,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if save_all:
            filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
            write_checkpoint(state, filename)
