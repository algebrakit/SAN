import torch
from tqdm import tqdm

from utils.utils import updata_lr, Meter, cal_score


def train(params, model, optimizer, epoch, train_loader, writer=None):

    model.train()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0
    loss_dt, word_loss_dt, struct_loss_dt, parent_loss_dt, kl_loss_dt = 0, 0, 0, 0, 0

    accumulation_steps = params.get('gradient_accumulation_steps', 1)
    scaler = params['scaler']

    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(pbar):

            if batch_idx % 100 == 0:
                # prevent OOM due to memory fragmentation
                torch.cuda.empty_cache()

            images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(device), label_masks.to(device)

            batch, time = labels.shape[:2]
            if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
                updata_lr(optimizer, epoch, batch_idx, len(train_loader), params['epoches'], params['lr'])

            # Only zero gradients at the start of accumulation
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=params.get('use_amp', False)):
                probs, loss = model(images, image_masks, labels, label_masks)

                word_loss, struct_loss, parent_loss, kl_loss = loss
                loss = (word_loss + struct_loss + parent_loss + kl_loss)

                # Scale loss by accumulation steps for averaging
                loss = loss / accumulation_steps

            loss_dt += loss.item()
            word_loss_dt += word_loss.item() / accumulation_steps
            struct_loss_dt += struct_loss.item() / accumulation_steps
            parent_loss_dt += parent_loss.item() / accumulation_steps
            kl_loss_dt += kl_loss.item() / accumulation_steps

            wordRate, structRate, ExpRate = cal_score(probs, labels, label_masks)

            word_right = word_right + wordRate * time
            struct_right = struct_right + structRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if params['gradient_clip']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])

                scaler.step(optimizer)
                scaler.update()

                loss_meter.add(loss_dt)

                if writer:
                    current_step = epoch * len(train_loader) + batch_idx + 1
                    writer.add_scalar('train/loss', loss_dt, current_step)
                    writer.add_scalar('train/word_loss', word_loss_dt, current_step)
                    writer.add_scalar('train/struct_loss', struct_loss_dt, current_step)
                    writer.add_scalar('train/WordRate', wordRate, current_step)
                    writer.add_scalar('train/parent_loss', parent_loss_dt, current_step)
                    writer.add_scalar('train/kl_loss', kl_loss_dt, current_step)
                    writer.add_scalar('train/structRate', structRate, current_step)
                    writer.add_scalar('train/ExpRate', ExpRate, current_step)
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_step)
                    writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch+1)
                    writer.add_scalar('epoch/train_WordRate', word_right / length, epoch+1)
                    writer.add_scalar('epoch/train_structRate', struct_right / length, epoch + 1)
                    writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, epoch + 1)

                pbar.set_description(f'Epoch: {epoch+1} LOSS: train: {loss_dt:.4f} parent: {parent_loss_dt:.4f} '
                                     f'KL: {kl_loss_dt:.4f} RATE: Word: {word_right / length:.4f}  '
                                     f'struct: {struct_right / length:.4f} Exp: {exp_right / cal_num:.4f}')
                # pbar.set_description(f'Epoch: {epoch+1} train loss: {loss_dt:.4f} word loss: {word_loss_dt:.4f} '
                #                      f'struct loss: {struct_loss_dt:.4f} parent loss: {parent_loss_dt:.4f} '
                #                      f'kl loss: {kl_loss_dt:.4f} WordRate: {word_right / length:.4f} '
                #                      f'structRate: {struct_right / length:.4f} ExpRate: {exp_right / cal_num:.4f}')

                loss_dt, word_loss_dt, struct_loss_dt, parent_loss_dt, kl_loss_dt = 0, 0, 0, 0, 0


        return loss_meter.mean, word_right / length, struct_right / length, exp_right / cal_num


def eval(params, model, epoch, eval_loader, writer=None):

    model.eval()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    with tqdm(eval_loader, total=len(eval_loader)) as pbar, torch.no_grad():

        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(eval_loader):

            images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(
                device), label_masks.to(device)

            batch, time = labels.shape[:2]

            # Mixed precision forward pass for evaluation
            with torch.amp.autocast('cuda', enabled=params.get('use_amp', False)):
                probs, loss = model(images, image_masks, labels, label_masks, is_train=False)

            word_loss, struct_loss = loss
            loss = word_loss + struct_loss
            loss_meter.add(loss.item())

            wordRate, structRate, ExpRate = cal_score(probs, labels, label_masks)

            word_right = word_right + wordRate * time
            struct_right = struct_right + structRate * time
            exp_right = exp_right + ExpRate
            length = length + time
            cal_num = cal_num + batch

            loss_dt = loss.item()
            word_loss_dt = word_loss.item()
            struct_loss_dt = struct_loss.item()

            if writer:
                current_step = epoch * len(eval_loader) + batch_idx + 1
                writer.add_scalar('eval/loss', loss_dt, current_step)
                writer.add_scalar('eval/word_loss', word_loss_dt, current_step)
                writer.add_scalar('eval/struct_loss', struct_loss_dt, current_step)
                writer.add_scalar('eval/WordRate', wordRate, current_step)
                writer.add_scalar('eval/structRate', structRate, current_step)
                writer.add_scalar('eval/ExpRate', ExpRate, current_step)

            pbar.set_description(f'Epoch: {epoch + 1} eval loss: {loss_dt:.4f} word loss: {word_loss_dt:.4f} '
                                 f'struct loss: {struct_loss_dt:.4f} WordRate: {word_right / length:.4f} '
                                 f'structRate: {struct_right / length:.4f} ExpRate: {exp_right / cal_num:.4f}')

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_structRate', struct_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / len(eval_loader.dataset), epoch + 1)
        return loss_meter.mean, word_right / length, struct_right / length, exp_right / cal_num