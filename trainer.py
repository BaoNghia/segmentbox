import time
import torch
import numpy as np
from tqdm import tqdm


def train_one_epoch(
        epoch, num_epoch,
        model, device,
        train_loader, valid_loader,
        criterion, optimizer,
        train_metrics, valid_metrics,
    ):
    # training-the-model
    with tqdm(enumerate(train_loader), total = len(train_loader)) as pbar:
        train_loss = 0
        mloss = 0
        # print("LR : {} \n".format(optimizer.param_groups[0]['lr']))
        model.train()
        for batch_i, (inputs, targets) in pbar:
            # move-tensors-to-GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward model
            outputs = model(inputs)
            # calculate-the-batch-loss
            loss = criterion(outputs, targets)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * inputs.size(0)
            ## calculate training metrics
            outputs = torch.sigmoid(outputs)
            train_metrics.step(outputs.cpu(), targets.cpu())

            ## ============= pbar =============
            mem = '%.3g GB' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            mloss = (mloss * batch_i + loss.item())/(batch_i + 1)
            s = ('%13s' * 2 + '%13.4g' * 1) % ('%g/%g' % (epoch, num_epoch-1), mem, mloss)
            pbar.set_description(s)
            pbar.set_postfix(Lr = optimizer.param_groups[0]['lr'])
        train_loss = train_loss/len(train_loader.dataset)
        
    #validate-the-model
    with tqdm(enumerate(valid_loader), total = len(valid_loader)) as pbar:
        valid_loss = 0
        pbar.set_description(('%26s'  + '%13s'* 1) % ('Train Loss', 'Val Loss'))
        model.eval()
        all_labels = []
        all_preds = []
        for batch_i, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # update-validation-loss
                valid_loss += loss.item() * inputs.size(0)
                outputs = torch.sigmoid(outputs)
            all_preds.append(outputs.cpu())
            all_labels.append(targets.cpu())

        valid_loss = valid_loss/len(valid_loader.dataset)
        all_preds = torch.cat(all_preds, axis=0)
        all_labels = torch.cat(all_labels, axis=0)
        valid_metrics.step(all_preds, all_labels)
        print(('%26.4g' + '%13.4g'* 1) % (train_loss, valid_loss))

    return (
        train_loss, valid_loss,
        train_metrics.epoch(),
        valid_metrics.last_step_metrics(),
    )