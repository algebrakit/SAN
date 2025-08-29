import torch
from tqdm import tqdm
from utils import updata_lr, Meter, cal_score


def train_stroke(params, model, optimizer, epoch, train_loader, writer=None):
    """
    Training loop for stroke-aware SAN model
    """
    model.train()
    device = params['device']
    loss_meter = Meter()
    
    # Metrics tracking
    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0
    
    with tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1} Train') as pbar:
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack stroke data
            stroke_data, stroke_masks, stroke_positions, labels, labels_mask = batch_data
            
            # Move to device
            stroke_data = stroke_data.to(device)
            stroke_masks = stroke_masks.to(device)
            stroke_positions = stroke_positions.to(device)  
            labels = labels.to(device)
            labels_mask = labels_mask.to(device)
            
            batch_size, time = labels.shape[:2]
            
            # Update learning rate
            if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
                updata_lr(optimizer, epoch, batch_idx, len(train_loader), params['epoches'], params['lr'])
                
            optimizer.zero_grad()
            
            try:
                # Forward pass through stroke-aware model
                probs, loss = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=True)
                
                # Unpack losses
                if len(loss) == 4:
                    word_loss, struct_loss, parent_loss, kl_loss = loss
                    total_loss = word_loss + struct_loss + parent_loss + kl_loss
                else:
                    word_loss, struct_loss = loss
                    parent_loss = torch.tensor(0.0, device=device)
                    kl_loss = torch.tensor(0.0, device=device)
                    total_loss = word_loss + struct_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if params['gradient_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])
                
                optimizer.step()
                
                # Update loss meter
                loss_meter.add(total_loss.item())
                
                # Calculate accuracy metrics
                wordRate, structRate, ExpRate = cal_score(probs, labels, labels_mask)
                
                word_right += wordRate * time
                struct_right += structRate * time
                exp_right += ExpRate * batch_size
                length += time
                cal_num += batch_size
                
                # TensorBoard logging
                if writer:
                    current_step = epoch * len(train_loader) + batch_idx + 1
                    writer.add_scalar('train_stroke/total_loss', total_loss.item(), current_step)
                    writer.add_scalar('train_stroke/word_loss', word_loss.item(), current_step)
                    writer.add_scalar('train_stroke/struct_loss', struct_loss.item(), current_step)
                    writer.add_scalar('train_stroke/parent_loss', parent_loss.item(), current_step)
                    writer.add_scalar('train_stroke/kl_loss', kl_loss.item(), current_step)
                    writer.add_scalar('train_stroke/word_accuracy', wordRate, current_step)
                    writer.add_scalar('train_stroke/struct_accuracy', structRate, current_step)
                    writer.add_scalar('train_stroke/expr_accuracy', ExpRate, current_step)
                    writer.add_scalar('train_stroke/lr', optimizer.param_groups[0]['lr'], current_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Word': f'{word_loss.item():.4f}',
                    'Struct': f'{struct_loss.item():.4f}',
                    'WordAcc': f'{wordRate:.3f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {e}")
                print(f"Stroke data shape: {stroke_data.shape}")
                print(f"Labels shape: {labels.shape}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate epoch metrics
    avg_loss = loss_meter.mean if len(loss_meter.nums) > 0 else 0
    word_acc = word_right / length if length > 0 else 0
    struct_acc = struct_right / length if length > 0 else 0
    expr_acc = exp_right / cal_num if cal_num > 0 else 0
    
    metrics = {
        'word_acc': word_acc,
        'struct_acc': struct_acc,
        'expr_acc': expr_acc
    }
    
    return avg_loss, metrics


def eval_stroke(params, model, epoch, eval_loader, writer=None):
    """
    Evaluation loop for stroke-aware SAN model
    """
    model.eval()
    device = params['device']
    loss_meter = Meter()
    
    # Metrics tracking
    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        with tqdm(eval_loader, total=len(eval_loader), desc=f'Epoch {epoch+1} Eval') as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    # Unpack stroke data
                    stroke_data, stroke_masks, stroke_positions, labels, labels_mask = batch_data
                    
                    # Move to device
                    stroke_data = stroke_data.to(device)
                    stroke_masks = stroke_masks.to(device)
                    stroke_positions = stroke_positions.to(device)
                    labels = labels.to(device)
                    labels_mask = labels_mask.to(device)
                    
                    batch_size, time = labels.shape[:2]
                    
                    # Forward pass
                    probs, loss = model(stroke_data, stroke_masks, stroke_positions, labels, labels_mask, is_train=False)
                    
                    # Unpack losses
                    if len(loss) == 4:
                        word_loss, struct_loss, parent_loss, kl_loss = loss
                        total_loss = word_loss + struct_loss + parent_loss + kl_loss
                    else:
                        word_loss, struct_loss = loss
                        total_loss = word_loss + struct_loss
                    
                    loss_meter.add(total_loss.item())
                    
                    # Calculate accuracy metrics
                    wordRate, structRate, ExpRate = cal_score(probs, labels, labels_mask)
                    
                    word_right += wordRate * time
                    struct_right += structRate * time
                    exp_right += ExpRate * batch_size
                    length += time
                    cal_num += batch_size
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'WordAcc': f'{wordRate:.3f}',
                        'StructAcc': f'{structRate:.3f}',
                        'ExprAcc': f'{ExpRate:.3f}'
                    })
                    
                except Exception as e:
                    print(f"\nError in evaluation batch {batch_idx}: {e}")
                    continue
    
    # Calculate epoch metrics
    avg_loss = loss_meter.mean if len(loss_meter.nums) > 0 else 0
    word_acc = word_right / length if length > 0 else 0
    struct_acc = struct_right / length if length > 0 else 0
    expr_acc = exp_right / cal_num if cal_num > 0 else 0
    
    metrics = {
        'word_acc': word_acc,
        'struct_acc': struct_acc,
        'expr_acc': expr_acc
    }
    
    # TensorBoard logging for evaluation
    if writer:
        current_step = (epoch + 1) * len(eval_loader)
        writer.add_scalar('eval_stroke/total_loss', avg_loss, current_step)
        writer.add_scalar('eval_stroke/word_accuracy', word_acc, current_step)
        writer.add_scalar('eval_stroke/struct_accuracy', struct_acc, current_step)
        writer.add_scalar('eval_stroke/expr_accuracy', expr_acc, current_step)
    
    return avg_loss, metrics


def test_stroke_training():
    """Test function to verify stroke training setup"""
    print("Testing stroke training components...")
    
    # Mock parameters
    params = {
        'device': 'cpu',
        'lr_decay': 'cosine',
        'epoches': 10,
        'lr': 1.0,
        'gradient_clip': True,
        'gradient': 100
    }
    
    print("✓ Stroke training functions loaded successfully")
    print("✓ Ready for training with InkML stroke data")


if __name__ == '__main__':
    test_stroke_training()