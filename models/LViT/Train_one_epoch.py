# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, mode, loss_pred, average_loss_pred, total_loss, average_total_loss, acc, average_acc, precision, average_precision, recall, average_recall, f1, average_f1, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'Regression Loss:{:.3f} '.format(loss_pred)
    string += '(Avg {:.4f}) '.format(average_loss_pred)
    string += 'Total Loss:{:.3f} '.format(total_loss)
    string += '(Avg {:.4f}) '.format(average_total_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Accuracy:{:.4f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    # string += 'Precision:{:.4f} '.format(precision)
    # string += '(Avg {:.4f}) '.format(average_precision)
    # string += 'Recall:{:.4f} '.format(recall)
    # string += '(Avg {:.4f}) '.format(average_recall)
    # string += 'F1:{:.4f} '.format(f1)
    # string += '(Avg {:.4f}) '.format(average_f1)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    # string += 'Regression Acc:{:.3f} '.format(r2)
    # string += '(Avg {:.4f}) '.format(average_r2)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary

from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import r2_score

# from sklearn.metrics import accuracy_score

# def regression_accuracy(preds, targets, tolerance=0.05):
#     preds_np = preds.cpu().numpy()
#     targets_np = targets.cpu().numpy()
    
#     # Discretizamos: 1 si están dentro del margen, 0 si no
#     correct = (np.abs(preds_np - targets_np) <= tolerance).astype(int)
#     return accuracy_score(correct, np.ones_like(correct))
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger, reg_criterion=torch.nn.MSELoss()):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum, loss_pred_sum, total_loss_sum, r2_sum, accuracy_sum, precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dices = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_ = BinaryAccuracy().to(device)
    precision_ = BinaryPrecision().to(device)
    recall_ = BinaryRecall().to(device)
    f1_ = BinaryF1Score().to(device)

    for i, (sampled_batch, names) in enumerate(loader, 1):
        # print('train one epoch')
        # print(i)
        # print(sampled_batch)
        # print(names)

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        # target es los días de supervivencia
        #print(sampled_batch.keys())
        images, masks, text, gt_value, ages = sampled_batch['image'], sampled_batch['label'], sampled_batch['text'], sampled_batch['target'],  sampled_batch['age']
        if text.shape[1] > 10:
            text = text[ :, :10, :]
        
        images, masks, text, gt_value, ages = images.cuda(non_blocking=True), masks.cuda(), text.cuda(), gt_value.cuda(), ages.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds, pred_value, _ = model(images, text, age=ages)

        # Pérdida de la predicción
        loss_pred = reg_criterion(pred_value, gt_value)

        alpha=0.5
        # La pérdida total es la de segmentación+MSE
        out_loss = criterion(preds, masks.float()) #+ alpha*loss_pred
        total_loss = criterion(preds, masks.float()) + loss_pred

        # Accuracy para segmentación y para regresión
        # accuracy_metric = BinaryAccuracy(threshold=0.5).to(images.device)
        # train_acc = accuracy_metric(preds, masks.int())
        # r2 = r2_score(gt_value.cpu().detach().numpy(), pred_value.cpu().detach().numpy())
        # r2 = balanced_accuracy_score(masks, preds)
        # y_pred = preds.int().view(-1)
        # y_true = masks.int().view(-1)
        # accuracy = accuracy_(y_pred, y_true).item()
        # precision = precision_(y_pred, y_true).item()
        # recall = recall_(y_pred, y_true).item()
        # f1 = f1_(y_pred, y_true).item()
        
        # print(model.training)

        if model.training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_dice = criterion._show_dice(preds, masks.float())
            train_iou = iou_on_batch(masks,preds)

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        # dices.append(train_dice)

        time_sum += len(images) * batch_time # time_sum.append(batch_time)
        loss_sum += len(images) * out_loss # loss_sum.append(out_loss)
        iou_sum += len(images) * train_iou # iou_sum.append(train_iou)
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice # dice_sum.append(train_dice)
        loss_pred_sum += len(images) * loss_pred
        total_loss_sum += len(images) * total_loss
        # acc_sum += len(images) * train_acc.item()
        # r2_sum += len(images) * r2
        # accuracy_sum += len(images) * accuracy
        # precision_sum += len(images) * precision
        # recall_sum += len(images) * recall
        # f1_sum += len(images)* f1

        # Promedio de las métricas por epoch
        if i == len(loader):
            average_loss_pred = loss_pred_sum / (config.batch_size*(i-1) + len(images))
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images)) # np.mean(loss_sum)
            average_time =  time_sum / (config.batch_size*(i-1) + len(images)) # np.mean(time_sum)
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images)) # np.mean(iou_sum)
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images)) # np.mean(dice_sum)
            average_total_loss = total_loss_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            # average_r2 = r2_sum / (config.batch_size*(i-1) + len(images))
            # average_acc = accuracy_sum / (config.batch_size*(i-1) + len(images))
            # average_precision = precision_sum / (config.batch_size*(i-1) + len(images))
            # average_recall = recall_sum / (config.batch_size*(i-1) + len(images))
            # average_f1 = f1_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss_pred = loss_pred_sum / (i * config.batch_size)
            average_loss = loss_sum / (i * config.batch_size) # np.mean(loss_sum)
            average_time = time_sum / (i * config.batch_size) # np.mean(time_sum)
            train_iou_average = iou_sum / (i * config.batch_size) # np.mean(iou_sum)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size) # np.mean(dice_sum)
            average_total_loss = total_loss_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            # average_r2 = r2_sum / (i * config.batch_size)
            # average_acc = accuracy_sum / (i * config.batch_size)
            # average_precision = precision_sum / (i * config.batch_size)
            # average_recall = recall_sum / (i * config.batch_size)
            # average_f1 = f1_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, logging_mode, loss_pred, average_loss_pred, total_loss, average_total_loss, 
                          acc = 0, 
                          average_acc=0, 
                          precision=0, 
                          average_precision=0, 
                          recall=0, 
                          average_recall=0, 
                          f1=0, 
                          average_f1=0,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)
            logger.info(f"Predicted value: {pred_value.detach().cpu().numpy()}")
            logger.info(f"GT value: {gt_value.detach().cpu().numpy()}")

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        del images, masks, text, gt_value, ages, preds, pred_value, loss_pred, out_loss, train_dice, train_iou
        if 'vis_path' in locals():
            del vis_path

        torch.cuda.empty_cache()
        import gc
        gc.collect()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg
