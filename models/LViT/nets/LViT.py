# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Aplicación avg pooling glogal--> Se aplica a las features profundas tras todo el encoder
        self.classifier_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten()
        )
        # Capa Lineal que predice un escalar, que son los días de supervivencia
        # self.classifier_head = nn.Linear(in_channels * 8 +512, 1)  # +1 por la edad, +512 por el texto
        # self.classifier_head = nn.Linear(1, 1)


        # Incremento del número de capas ocultas para la regresión
        # self.classifier_head = nn.Sequential(
        #     # nn.Linear(in_channels * 8 + 1, 128), Aquí se concatena la edad
        #     nn.Linear(in_channels * 8, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )
        self.classifier_head = nn.Sequential(
            nn.Linear(1, 128),  # 513 si text4 tiene 512 canales; y4 es 1
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.classifier_head = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )



        # Variar el porcentaje de Dropout
        # self.classifier_head = nn.Sequential(
        #     nn.Linear(1, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),  # Dropout del 50%
        #     nn.Linear(128, 1)
        # )

        # Aumentar el número de neuronas en la capa oculta actual
        # self.classifier_head = nn.Sequential(
        #     # nn.Linear(in_channels * 8, 512),  # más neuronas
        #     nn.Linear(1, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )

        # Cabeza de regresión inspirada en Gradient Boosting
        # self.classifier_head = nn.Sequential(
        #     nn.LayerNorm(512),              # Normaliza los embeddings como un preprocesado robusto
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),                # Regularización para evitar overfitting
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)                # Salida final escalar
        # )

        # Cabezas nuevas de regresión
        # self.classifier_head = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        # self.classifier_head = nn.Sequential(
        #     nn.LayerNorm(512),         # Normalización de entrada
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)             # No hay activación final (para salida continua)
        # )



    # Concatenación de la imagen con el texto procesado
    # def forward(self, x, text, age=None):
    #     x = x.float()
    #     x1 = self.inc(x)
    #     text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)
    #     text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
    #     text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
    #     text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
    #     y1 = self.downVit(x1, x1, text1)
    #     x2 = self.down1(x1)
    #     y2 = self.downVit1(x2, y1, text2)
    #     x3 = self.down2(x2)
    #     y3 = self.downVit2(x3, y2, text3)
    #     x4 = self.down3(x3)
    #     y4 = self.downVit3(x4, y3, text4)
    #     x5 = self.down4(x4)

    #     # Pool imagen
    #     class_input = self.classifier_pool(x5)

    #     # Pool texto
    #     pooled_text4 = F.adaptive_avg_pool2d(text4, (1, 1))
    #     pooled_text4 = pooled_text4.view(pooled_text4.size(0), -1)

    #     # Concatenar imagen + texto
    #     concat_features = torch.cat([class_input, pooled_text4], dim=1)  # [batch_size, features_total]

    #     class_pred = self.classifier_head(concat_features)

    #     # reconstrucción para segmentación
    #     y4 = self.upVit3(y4, y4, text4, True)
    #     y3 = self.upVit2(y3, y4, text3, True)
    #     y2 = self.upVit1(y2, y3, text2, True)
    #     y1 = self.upVit(y1, y2, text1, True)
    #     x1 = self.reconstruct1(y1) + x1
    #     x2 = self.reconstruct2(y2) + x2
    #     x3 = self.reconstruct3(y3) + x3
    #     x4 = self.reconstruct4(y4) + x4
    #     x = self.up4(x5, x4)
    #     x = self.up3(x, x3)
    #     x = self.up2(x, x2)
    #     x = self.up1(x, x1)

    #     if self.n_classes == 1:
    #         logits = self.last_activation(self.outc(x))
    #     else:
    #         logits = self.outc(x)

    #     return logits, class_pred

    # Usando la salida del bloque de vision transformer (y4)
    def forward(self, x, text, age=None):
        x = x.float()  # imagen de entrada
        x1 = self.inc(x)
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        y1 = self.downVit(x1, x1, text1)
        x2 = self.down1(x1)
        y2 = self.downVit1(x2, y1, text2)
        x3 = self.down2(x2)
        y3 = self.downVit2(x3, y2, text3)
        x4 = self.down3(x3)
        y4 = self.downVit3(x4, y3, text4)
        x5 = self.down4(x4)

        # Para regresión usamos directamente y4
        # print(y4.shape)
        # pooled_y4 = y4.mean(dim=1)  # Promedia sobre los 196 patches → [B, 512]
        # class_pred = self.classifier_head(pooled_y4)
        y4_encoder = y4
        pooled_y4 = y4.mean(dim=1)

        pooled_y4 = F.adaptive_avg_pool2d(y4, (1, 1))  # [batch, channels, 1, 1]
        pooled_y4 = pooled_y4.view(pooled_y4.size(0), -1)  # [batch, channels]
        class_pred = self.classifier_head(pooled_y4)

        # Luego sigue tu reconstrucción para segmentación
        y4 = self.upVit3(y4, y4, text4, True)
        y3 = self.upVit2(y3, y4, text3, True)
        y2 = self.upVit1(y2, y3, text2, True)
        y1 = self.upVit(y1, y2, text1, True)
        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)

        return logits, class_pred, y4_encoder


    # def forward(self, x, text, age=None):
    #     x = x.float()  # x [4,3,224,224]
    #     x1 = self.inc(x)  # x1 [4, 64, 224, 224]
    #     text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
    #     text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
    #     text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
    #     text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
    #     y1 = self.downVit(x1, x1, text1)
    #     x2 = self.down1(x1)
    #     y2 = self.downVit1(x2, y1, text2)
    #     x3 = self.down2(x2)
    #     y3 = self.downVit2(x3, y2, text3)
    #     x4 = self.down3(x3)
    #     y4 = self.downVit3(x4, y3, text4)
    #     x5 = self.down4(x4)

    #     # Para PREDICCIÓN de los días de supervivencia
    #     # Le aplicamos una capa de avg pooling global a las features profundas resultantes de haber pasado por todo el encoder
    #     class_input = self.classifier_pool(x5)

    #     # Le concatenamos a lo de antes la edad del paciente
    #     # if age is not None:
    #     #   #class_input = torch.cat([class_input, age.unsqueeze(1)], dim=1)
    #     #   if age.dim() == 1:
    #     #       age = age.unsqueeze(1)  # shape: [batch_size, 1]
    #     #   elif age.dim() == 3:
    #     #       age = age.squeeze()  # en caso de que venga como [batch_size, 1, 1]
    #     #   class_input = torch.cat([class_input, age/100.0], dim=1)

    #     # else:
    #     #   batch_size = class_input.size(0)
    #     #   zero_age = torch.zeros(batch_size, 1, device=class_input.device)
    #     #   class_input = torch.cat([class_input, zero_age], dim=1)
        
    #     # Pasamos esa concatenación por una capa linear que predice un escalar
    #     class_pred = self.classifier_head(class_input)

    #     y4 = self.upVit3(y4, y4, text4, True)
    #     y3 = self.upVit2(y3, y4, text3, True)
    #     y2 = self.upVit1(y2, y3, text2, True)
    #     y1 = self.upVit(y1, y2, text1, True)
    #     x1 = self.reconstruct1(y1) + x1
    #     x2 = self.reconstruct2(y2) + x2
    #     x3 = self.reconstruct3(y3) + x3
    #     x4 = self.reconstruct4(y4) + x4
    #     x = self.up4(x5, x4)
    #     x = self.up3(x, x3)
    #     x = self.up2(x, x2)
    #     x = self.up1(x, x1)
    #     if self.n_classes == 1:
    #         logits = self.last_activation(self.outc(x))
    #     else:
    #         logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
    #     return logits, class_pred
