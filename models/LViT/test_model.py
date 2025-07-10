import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.LViT import LViT
from utils import *
import cv2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred


# def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
#     model.eval()

#     output = model(input_img.cuda(), text.cuda())
#     pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
#     predict_save = pred_class[0].cpu().data.numpy()
#     predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
#     dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
#                                                   save_path=vis_save_path + '_predict' + model_type + '.jpg')
#     return dice_pred_tmp, iou_tmp

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

# def vis_and_save_heatmap(model, input_img, text, age, labs, vis_save_path, dice_pred, dice_ens, ground_truth_sd):
#     model.eval()
#     output, class_pred = model(input_img.cuda(), text.cuda(), age.cuda())
    
#     # REGRESIÓN: Evaluar predicción de días de supervivencia
#     pred_sd = class_pred.detach().cpu().numpy().flatten()
#     gt_sd = ground_truth_sd.cpu().numpy().flatten()

#     mae = mean_absolute_error(gt_sd, pred_sd)
#     mse = mean_squared_error(gt_sd, pred_sd)
#     rmse = np.sqrt(mse)
#     #rmse = mean_squared_error(gt_sd, pred_sd, squared=False)
#     r2 = r2_score(gt_sd, pred_sd)

#     print(f"Pred: {pred_sd}, GT: {gt_sd}")
#     print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

#     # SEGMENTACIÓN: calcular Dice e IoU
#     pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
#     predict_save = pred_class[0].cpu().data.numpy()
#     predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
#     dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path + '_predict' + model_type + '.jpg')
    
#     return dice_pred_tmp, iou_tmp, mae, rmse, r2

def vis_and_save_heatmap(model, input_img, text, age, labs, vis_save_path, dice_pred, dice_ens, ground_truth_sd):
    model.eval()
    output, class_pred, _ = model(input_img.cuda(), text.cuda(), age.cuda())
    
    # REGRESIÓN: Evaluar predicción de días de supervivencia
    pred_sd = class_pred.detach().cpu().numpy().flatten()
    gt_sd = ground_truth_sd.cpu().numpy().flatten()

    # MAE normalizado
    mae = metric(gt_sd, pred_sd)
    mse = mean_squared_error(gt_sd, pred_sd)
    rmse = np.sqrt(mse)
    r2 = r2_score(gt_sd, pred_sd)

    print(f"Pred: {pred_sd}, GT: {gt_sd}")
    print(f"MAE normalizado: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # SEGMENTACIÓN: calcular Dice e IoU
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path + '_predict' + model_type + '.jpg')
    
    return dice_pred_tmp, iou_tmp, mae, rmse, r2

import csv

def evaluate_and_save(model, loader, save_csv_path, vis_path_prefix):
    model.eval()
    dice_total, iou_total = 0.0, 0.0
    results = []

    for i, (sampled_batch, names) in enumerate(loader, 1):
        test_data = sampled_batch['image']
        test_label = sampled_batch['label']
        test_text = sampled_batch['text']
        test_age = sampled_batch['age']
        test_target = sampled_batch['target']

        arr = test_data.numpy().astype(np.float32)
        lab = test_label.data.numpy()
        input_img = torch.from_numpy(arr)

        # Inferencia
        dice_pred_t, iou_pred_t, mae, rmse, r2 = vis_and_save_heatmap(
            model, input_img, test_text, test_age, lab,
            vis_path_prefix + str(names),
            dice_pred=0.0, dice_ens=0.0,
            ground_truth_sd=test_target
        )

        # Predicción numérica para SD
        pred_sd = float(
            model(input_img.cuda(), test_text.cuda(), test_age.cuda())[1]
            .detach().cpu().numpy().flatten()[0]
        )

        results.append({
            'name': str(names[0]),
            'dice': dice_pred_t,
            'iou': iou_pred_t,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'gt_age': test_target.item(),
            'pred_age': pred_sd
        })

        dice_total += dice_pred_t
        iou_total += iou_pred_t

    # Guardar en CSV
    keys = results[0].keys()
    with open(save_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"[{save_csv_path}] Dice: {dice_total / len(loader):.4f}, IoU: {iou_total / len(loader):.4f}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    model_type = config.model_name
    model_path = f"./{config.task_name}/{model_type}/{test_session}/models/best_model-{model_type}.pth.tar"
    save_path = f"./{config.task_name}/{model_type}/{test_session}/"
    vis_path = f"./{config.task_name}_visualize_test/"

    os.makedirs(vis_path, exist_ok=True)

    checkpoint = torch.load(model_path, map_location='cuda')

    config_vit = config.get_CTranS_config()
    model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded!')

    # Transforms
    tf_transform = ValGenerator(output_size=[config.img_size, config.img_size])

    # Load datasets
    train_text = read_text_brats2020(config.train_dataset + 'Train_text.xlsx')
    val_text = read_text_brats2020(config.val_dataset + 'Val_text.xlsx')
    test_text = read_text_brats2020(config.test_dataset + 'Test_text.xlsx')

    train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, tf_transform, image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, tf_transform, image_size=config.img_size)
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_transform, image_size=config.img_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluación
    evaluate_and_save(model, train_loader, f"{save_path}/predictions_train.csv", vis_path + "train_")
    evaluate_and_save(model, val_loader, f"{save_path}/predictions_val.csv", vis_path + "val_")
    evaluate_and_save(model, test_loader, f"{save_path}/predictions_test.csv", vis_path + "test_")


# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     test_session = config.test_session

#     if config.task_name == "MoNuSeg":
#         test_num = 14
#         model_type = config.model_name
#         model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

#     elif config.task_name == "Covid19":
#         test_num = 2113
#         model_type = config.model_name
#         model_path = "./Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

#     elif config.task_name == "Processed_BraTS2020":
#         test_num = 38 # 134 # 38 # 8
#         model_type = config.model_name
#         model_path = "./Processed_BraTS2020/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    
#     save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
#     vis_path = "./" + config.task_name + '_visualize_test/'
#     if not os.path.exists(vis_path):
#         os.makedirs(vis_path)

#     checkpoint = torch.load(model_path, map_location='cuda')

#     if model_type == 'LViT':
#         config_vit = config.get_CTranS_config()
#         model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

#     elif model_type == 'LViT_pretrain':
#         config_vit = config.get_CTranS_config()
#         model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


#     else:
#         raise TypeError('Please enter a valid name for the model type')

#     model = model.cuda()
#     if torch.cuda.device_count() > 1:
#        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
#        model = nn.DataParallel(model)
#     model.load_state_dict(checkpoint['state_dict'], strict=False)
#     print('Model loaded !')
#     tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
#     #test_text = read_text(config.test_dataset + 'Test_text.xlsx')
#     # Para el conjunto de test
#     test_text = read_text_brats2020(config.test_dataset + 'Test_text.xlsx')
#     test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     # Para el cojunto de train y validación
#     train_text = read_text_brats2020(config.train_dataset + 'Train_text.xlsx')
#     val_text = read_text_brats2020(config.val_dataset + 'Val_text.xlsx')

#     train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, tf_test, image_size=config.img_size)
#     val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, tf_test, image_size=config.img_size)

#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


#     dice_pred = 0.0
#     iou_pred = 0.0
#     dice_ens = 0.0

#     with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
#         for i, (sampled_batch, names) in enumerate(test_loader, 1):
#             # print(names)
#             test_data, test_label, test_text, test_age, test_target = sampled_batch['image'], sampled_batch['label'], sampled_batch['text'], sampled_batch['age'], sampled_batch['target']
#             arr = test_data.numpy()
#             arr = arr.astype(np.float32())
#             lab = test_label.data.numpy()
#             img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
#             fig, ax = plt.subplots()
#             plt.imshow(img_lab, cmap='gray')
#             plt.axis("off")
#             height, width = config.img_size, config.img_size
#             fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
#             plt.margins(0, 0)
#             plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
#             plt.close()
#             input_img = torch.from_numpy(arr)
#             # dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, test_text, None, lab,
#             #                                                vis_path + str(names),
#             #                                                dice_pred=dice_pred, dice_ens=dice_ens)
#             # model, input_img, text, age, img_RGB, labs, vis_save_path, dice_pred, dice_ens, ground_truth_sd
#             dice_pred_t, iou_pred_t, mae, rmse, r2 = vis_and_save_heatmap(model, input_img, test_text, test_age, lab,
#                                                                           vis_path + str(names),
#                                                                           dice_pred=dice_pred, dice_ens=dice_ens,
#                                                                           ground_truth_sd=test_target)

            
#             dice_pred += dice_pred_t
#             iou_pred += iou_pred_t
#             torch.cuda.empty_cache()
#             pbar.update()
#     print("dice_pred", dice_pred / test_num)
#     print("iou_pred", iou_pred / test_num)
