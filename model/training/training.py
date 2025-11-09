import torch
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.utils.utils import updata_lr, Meter, cal_score


def get_batch_memory_info(images, image_masks, batch):
    """
    Compute batch characteristics for bucket identification.

    Returns formatted string with: batch size, max pixels, avg pixels, effective memory load
    """
    batch_heights = []
    batch_widths = []

    for i in range(batch):
        # Find actual image dimensions (non-zero regions in mask)
        h_mask = image_masks[i, 0, :, 0].sum()  # Height used
        w_mask = image_masks[i, 0, 0, :].sum()  # Width used
        if h_mask > 0:
            batch_heights.append(int(h_mask.item()))
        if w_mask > 0:
            batch_widths.append(int(w_mask.item()))

    if batch_heights and batch_widths:
        max_pixels = max(h * w for h, w in zip(batch_heights, batch_widths))
        avg_pixels = sum(h * w for h, w in zip(batch_heights, batch_widths)) / len(batch_heights)
        effective_load = batch * avg_pixels  # Effective memory load
    else:
        max_pixels = 0
        avg_pixels = 0
        effective_load = 0

    return f"Batch: {batch} samples, Max pixels: {max_pixels:,}, Avg pixels: {avg_pixels:,.0f}, Effective load: {effective_load:,.0f}"


def train(params, model, optimizer, epoch, train_loader, writer=None):

    model.train()
    device = params['device']
    loss_meter = Meter()

    total_words_right, total_structs_right, total_exp_right, length, cal_num = 0, 0, 0, 0, 0
    loss_dt, word_loss_dt, struct_loss_dt, parent_loss_dt, kl_loss_dt, eos_penalty_dt = 0, 0, 0, 0, 0, 0

    # Sample-based accumulation instead of batch-based
    min_samples_per_step = params.get('min_samples_per_step', 18)  # Default: base_batch_size × 3
    accumulated_samples = 0
    accumulated_lines = 0  # Track total tokens (batch × time) for proper gradient normalization
    scaler = params['scaler']
    epoch_step_count = 0  # Track optimizer steps within this epoch (for LR schedule)

    # Estimate total optimizer steps per epoch for LR schedule
    # This is approximate but close enough for cosine scheduling
    estimated_steps_per_epoch = max(1, len(train_loader.dataset) // min_samples_per_step)

    # Global step counter for TensorBoard (persists across epochs)
    global_step = epoch * estimated_steps_per_epoch

    # Track total samples processed this epoch for validation
    total_samples_processed = 0

    # Log LR schedule information at epoch start for verification
    if epoch == 0:
        print(f"\n📊 Learning Rate Schedule Info:")
        print(f"   Dataset size: {len(train_loader.dataset)} samples")
        print(f"   Min samples per step: {min_samples_per_step}")
        print(f"   Estimated optimizer steps per epoch: {estimated_steps_per_epoch}")
        print(f"   Initial LR: {params['lr']}")
        print(f"   LR decay: {params.get('lr_decay', 'cosine')}")
        print(f"   Total epochs: {params['epoches']}")

    peak_mem = 0

    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(pbar):
            # image_masks: [batch_index, 0, height, width]
            # label_masks: [batch index, line index, (1 if line applies, 1 if line is struct)]

            if epoch_step_count > 0 and epoch_step_count % 100 == 0:
                # prevent OOM due to memory fragmentation (clear based on optimizer steps, not batches)
                torch.cuda.empty_cache()

            images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(device), label_masks.to(device)

            batch = labels.shape[0]

            # Only zero gradients when starting fresh accumulation
            if accumulated_samples == 0:
                optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=params.get('use_amp', False)):
                probs, loss_components = model(images, image_masks, labels, label_masks)

                # loss is aggregated over all lines in hybrid tree and all items in batch (not averaged)
                word_loss, struct_loss, parent_loss, kl_loss, eos_penalty = loss_components
                loss = word_loss + struct_loss + parent_loss + kl_loss + eos_penalty

            # Track losses (unnormalized)
            loss_dt += loss.item()
            word_loss_dt += word_loss.item()
            struct_loss_dt += struct_loss.item()
            eos_penalty_dt += eos_penalty.item()
            if params['decoder']['inverse']:
                parent_loss_dt += parent_loss.item()
                kl_loss_dt += kl_loss.item()

            words_right, structs_right, exp_right = cal_score(probs, labels, label_masks)

            valid_lines = label_masks[:,:,0].sum().item()  # Count only valid (non-padding) label lines (hybrid tree lines)
            total_words_right = total_words_right + words_right
            total_structs_right = total_structs_right + structs_right
            total_exp_right = total_exp_right + exp_right 
            length = length + valid_lines  # Count valid lines up to this batch
            cal_num = cal_num + batch      # Total number of samples up to this batch

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Track accumulated samples and tokens
            accumulated_samples += batch
            accumulated_lines += valid_lines  # Count only valid (non-padding) label lines (hybrid tree lines)
            total_samples_processed += batch

            mem_use = torch.cuda.max_memory_allocated() / 1e9
            if mem_use > peak_mem:
                peak_mem = mem_use
                batch_info = get_batch_memory_info(images, image_masks, batch)
                print(f"Peak memory: {peak_mem:.2f} GB | {batch_info}")
            if batch_idx % 3000 == 0:
                batch_info = get_batch_memory_info(images, image_masks, batch)
                print(f"Peak memory: {peak_mem:.2f} GB | {batch_info}")

            # Step optimizer when we've accumulated enough samples OR at end of epoch (if configured)
            is_last_batch = (batch_idx + 1) == len(train_loader)
            should_step = accumulated_samples >= min_samples_per_step or is_last_batch

            if should_step:
                # Update learning rate (moved here to align with optimizer steps, not batches)
                if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
                    # Use epoch_step_count (0-indexed within epoch) and estimated_steps_per_epoch
                    updata_lr(optimizer, epoch, epoch_step_count, estimated_steps_per_epoch, params['epoches'], params['lr'])

                # Correct gradient scaling order for AMP + accumulation
                # 1. First unscale from AMP
                if params.get('use_amp', False):
                    scaler.unscale_(optimizer)

                # 2. Then normalize gradients by total accumulated tokens (batch × time)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(accumulated_lines)

                # 3. Finally clip gradients (on properly scaled gradients)
                if params['gradient_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])

                scaler.step(optimizer)
                scaler.update()
                epoch_step_count += 1  # Increment per-epoch step counter
                global_step += 1  # Increment global step counter for TensorBoard

                # Normalize losses by accumulated tokens for logging (matches gradient normalization)
                loss_dt_normalized = loss_dt / accumulated_lines
                word_loss_dt_normalized = word_loss_dt / accumulated_lines
                struct_loss_dt_normalized = struct_loss_dt / accumulated_lines
                parent_loss_dt_normalized = parent_loss_dt / accumulated_lines if params['decoder']['inverse'] else 0
                kl_loss_dt_normalized = kl_loss_dt / accumulated_lines if params['decoder']['inverse'] else 0
                eos_penalty_dt_normalized = eos_penalty_dt / accumulated_lines if params['decoder'].get('eos_penalty', False) else 0

                loss_meter.add(loss_dt_normalized)

                if writer:
                    # Use global_step for consistent TensorBoard x-axis (continuous across epochs)
                    writer.add_scalar('train/loss', loss_dt_normalized, global_step)
                    writer.add_scalar('train/word_loss', word_loss_dt_normalized, global_step)
                    writer.add_scalar('train/struct_loss', struct_loss_dt_normalized, global_step)
                    writer.add_scalar('train/WordRate', words_right / accumulated_lines, global_step)
                    writer.add_scalar('train/parent_loss', parent_loss_dt_normalized, global_step)
                    writer.add_scalar('train/kl_loss', kl_loss_dt_normalized, global_step)
                    writer.add_scalar('train/eos_penalty', eos_penalty_dt_normalized, global_step)
                    writer.add_scalar('train/structRate', structs_right / accumulated_lines / 7, global_step)
                    writer.add_scalar('train/ExpRate', exp_right / accumulated_samples, global_step)
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('train/accumulated_samples', accumulated_samples, global_step)  # Track actual samples per step
                    writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch+1)
                    writer.add_scalar('epoch/train_WordRate', total_words_right / length, epoch+1)
                    writer.add_scalar('epoch/train_structRate', total_structs_right / length / 7, epoch + 1)
                    writer.add_scalar('epoch/train_ExpRate', total_exp_right / cal_num, epoch + 1)

                pbar.set_description(f'Epoch: {epoch+1} LOSS: train: {loss_dt_normalized:.4f} '
                                     f'RATE: Word: {total_words_right / length:.4f}  '
                                     f'struct: {total_structs_right / length / 7:.4f} Exp: {total_exp_right / cal_num:.4f} '
                                     f'samples: {accumulated_samples}')

                # Reset accumulators
                loss_dt, word_loss_dt, struct_loss_dt, parent_loss_dt, kl_loss_dt, eos_penalty_dt = 0, 0, 0, 0, 0, 0
                accumulated_samples = 0
                accumulated_lines = 0

        # Validation logging: verify all samples were processed
        dataset_size = len(train_loader.dataset)
        if total_samples_processed != dataset_size:
            print(f"\n⚠️  WARNING: Sample count mismatch!")
            print(f"   Expected: {dataset_size} samples")
            print(f"   Processed: {total_samples_processed} samples")
            print(f"   Difference: {dataset_size - total_samples_processed} samples")
        else:
            print(f"\n✓ Epoch {epoch+1} complete: {total_samples_processed}/{dataset_size} samples processed ({epoch_step_count} optimizer steps)")

        return loss_meter.mean, total_words_right / length, total_structs_right / length / 7, total_exp_right / cal_num


def eval(params, model, epoch, eval_loader, writer=None):

    model.eval()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0
    total_loss, total_word_loss, total_struct_loss, total_valid_tokens = 0, 0, 0, 0

    with tqdm(eval_loader, total=len(eval_loader)) as pbar, torch.no_grad():

        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(eval_loader):

            images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(
                device), label_masks.to(device)

            batch = labels.shape[0]

            # Mixed precision forward pass for evaluation
            with torch.amp.autocast('cuda', enabled=params.get('use_amp', False)):
                probs, loss = model(images, image_masks, labels, label_masks, is_train=False)

            word_loss, struct_loss = loss
            loss = word_loss + struct_loss

            words_right, structs_right, numExpCorrect = cal_score(probs, labels, label_masks)

            valid_tokens = label_masks[:,:,0].sum().item()
            word_right = word_right + words_right      # ✓ Just accumulate counts
            struct_right = struct_right + structs_right  # ✓ Just accumulate counts

            exp_right = exp_right + numExpCorrect
            length = length + valid_tokens  # Only count valid tokens
            cal_num = cal_num + batch

            # Accumulate raw losses (sums over tokens)
            total_loss += loss.item()
            total_word_loss += word_loss.item()
            total_struct_loss += struct_loss.item()
            total_valid_tokens += valid_tokens

            # Normalize losses by valid tokens in this batch for display
            loss_dt = loss.item() / valid_tokens
            word_loss_dt = word_loss.item() / valid_tokens
            struct_loss_dt = struct_loss.item() / valid_tokens

            loss_meter.add(loss_dt)

            if writer:
                current_step = epoch * len(eval_loader) + batch_idx + 1
                writer.add_scalar('eval/loss', loss_dt, current_step)
                writer.add_scalar('eval/word_loss', word_loss_dt, current_step)
                writer.add_scalar('eval/struct_loss', struct_loss_dt, current_step)
                writer.add_scalar('eval/WordRate', word_right/length, current_step)
                writer.add_scalar('eval/structRate', struct_right / length / 7, current_step)
                writer.add_scalar('eval/ExpRate', numExpCorrect, current_step)

            pbar.set_description(f'Epoch: {epoch + 1} eval loss: {loss_dt:.4f} word loss: {word_loss_dt:.4f} '
                                 f'struct loss: {struct_loss_dt:.4f} WordRate: {word_right / length:.4f} '
                                 f'structRate: {struct_right / length / 7:.4f} ExpRate: {exp_right / cal_num:.4f}')

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_structRate', struct_right / length / 7, epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / cal_num, epoch + 1)
        return loss_meter.mean, word_right / length, struct_right / length / 7, exp_right / cal_num
