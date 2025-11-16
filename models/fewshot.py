import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Res101Encoder, Res50Encoder
from .decoder_p import MSA_head
from .deconder import EnhancedDecoder, SemanticConsistencyFusion


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        # self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
        #                              pretrained_weights=pretrained_weights)  # or "resnet101"
        self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"

        self.scaler = 20.0
        self.iter = 1
        self.device = torch.device('cuda')
        self.criterion = nn.NLLLoss()
        self.decoder1 = EnhancedDecoder()
        ###
        self.B_TA = MSA_head(dim=256)
        self.F_TA = MSA_head(dim=256)
        self.TA = MSA_head(dim=256)
        # self.Fuse = ResidualGatedFusion(in_channels=3 * 256, out_channels=256)
        self.Fuse = nn.Conv2d(3 * 256, 256, kernel_size=3, padding=1)
        ##############
        self.reduceconv4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension
        self.reduceconv3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension
        self.reduceconv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension
        self.reduceconv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension

        self.cdcf = SemanticConsistencyFusion({
            "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048
        })

        ##########duibi
        self.conv1 = nn.Conv2d(768, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # torch.Size([1, 1, 1, 256, 256])

        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # torch.Size([3, 3, 256, 256])

        img_fts, tao = self.encoder(imgs_concat[0:2])
        # Get threshold #
        self.t = tao[1:2]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        qry_fts_out = None
        if train:
            img_fts1, _ = self.encoder(imgs_concat[2:3])
            ##################################20250704
            seg_fts_query = {k: v[1:2] for k, v in img_fts.items()}
            fused_fts = self.cdcf(seg_fts_query, img_fts1)
            for layer_name in fused_fts:
                feat_clone = img_fts[layer_name].clone()  # [2, C, H, W]
                feat_clone[1] = fused_fts[layer_name].squeeze(0)  # [C, H, W]
                img_fts[layer_name] = feat_clone
            ######################################
            t = []
            t.append(img_fts1["layer1"])
            t.append(img_fts1["layer2"])
            t.append(img_fts1["layer3"])
            t.append(img_fts1["layer4"])
            qry_fts_out = self.decoder1(t)  # torch.Size([1, 1, 256, 256])

        outputs = []
        align_loss = torch.zeros(1).to(self.device)

        ############
        # layer4_fts = self.reduceconv4(img_fts["layer4"])  # torch.Size([2, 1024, 32, 32])
        layer3_fts = self.reduceconv3(img_fts["layer3"])  # torch.Size([2, 512, 64, 64])
        layer2_fts = self.reduceconv2(img_fts["layer2"])  # torch.Size([2, 256, 64, 64])
        layer1_fts = self.reduceconv1(img_fts["layer1"])  # torch.Size([2, 256, 64, 64])

        # supp_fts_4 = self.getFeatures(layer4_fts[0:1], supp_mask[0][0])  # torch.Size([1, 1024])
        # fg_prototypes_4 = self.getPrototype([[supp_fts_4]])  # prototype for support  list 1 torch.Size([1, 1024])

        supp_fts_3 = self.getFeatures(layer3_fts[0:1], supp_mask[0][0])  # torch.Size([1, 256])
        fg_prototypes_3 = self.getPrototype([[supp_fts_3]])  # prototype for support list[torch.Size([1, 256])]
        ########
        supp_fts_3b = self.getFeatures(layer3_fts[0:1], 1 - supp_mask[0][0])  # torch.Size([1, 256])
        fg_prototypes_3b = self.getPrototype([[supp_fts_3b]])  # list[torch.Size([1, 256])]

        supp_fts_2 = self.getFeatures(layer2_fts[0:1], supp_mask[0][0])  # torch.Size([1, 256])
        fg_prototypes_2 = self.getPrototype([[supp_fts_2]])  # list[torch.Size([1, 256])]
        ########
        supp_fts_2b = self.getFeatures(layer2_fts[0:1], 1 - supp_mask[0][0])  # torch.Size([1, 256])
        fg_prototypes_2b = self.getPrototype([[supp_fts_2b]])  # list[torch.Size([1, 256])]

        supp_fts_1 = self.getFeatures(layer1_fts[0:1], supp_mask[0][0])  # torch.Size([1, 256])
        fg_prototypes_1 = self.getPrototype([[supp_fts_1]])  # list[torch.Size([1, 256])]
        ########
        supp_fts_1b = self.getFeatures(layer1_fts[0:1], 1 - supp_mask[0][0])  # torch.Size([1, 256])
        fg_prototypes_1b = self.getPrototype([[supp_fts_1b]])  # list[torch.Size([1, 256])]

        qry_pred_1 = torch.stack(
            [self.getPred(layer3_fts[1:2], fg_prototypes_3[0], self.thresh_pred[way])
             for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'  torch.Size([1, 1, 64, 64])

        qry_pred_up = F.interpolate(qry_pred_1, size=img_size, mode='bilinear', align_corners=True)  # torch.Size([1, 1, 256, 256])
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        if train:
            align_loss_epi = self.alignLoss(layer3_fts[0:1].unsqueeze(0), layer3_fts[1:2], preds, supp_mask[0])
            align_loss += align_loss_epi

        qry_fts = layer3_fts[1:2]
        mask_prob_detached = qry_pred_1.detach()
        bd_mask, _ = self.make_boundary_mask(mask_prob_detached)
        N, C, H, W = qry_fts.shape
        xf = self.F_TA(qry_fts, mask_prob_detached)
        xb = self.B_TA(qry_fts, 1 - mask_prob_detached)
        x = self.TA(qry_fts, bd_mask)
        x = torch.cat((xb, xf, x), 1)
        x = x.view(N, 3 * C, H, W)
        qry_fts3 = self.Fuse(x)

        #################bijiao
        _, _, h, w = qry_fts3.shape
        b_expanded = fg_prototypes_3b[0].unsqueeze(-1).unsqueeze(-1)  # [1, 256, 1, 1]
        b_expanded = b_expanded.expand(-1, -1, h, w)  # [1, 256, h, w]
        f_expanded = fg_prototypes_3[0].unsqueeze(-1).unsqueeze(-1)  # [1, 256, 1, 1]
        f_expanded = f_expanded.expand(-1, -1, h, w)  # [1, 256, h, w]
        combined = torch.cat([qry_fts3, b_expanded, f_expanded], dim=1)  # [1, 768, h, w]
        x = self.conv1(combined)
        x = self.conv2(x)
        qry_pred_2 = self.sigmoid(x)
        #################bijiao

        qry_pred_2 = qry_pred_2 + qry_pred_1

        qry_pred_up = F.interpolate(qry_pred_2, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        if train:
            align_loss_epi = self.alignLoss_bijiao(layer3_fts[0:1].unsqueeze(0), qry_fts3, preds, supp_mask[0])
            align_loss = align_loss + 0.3 * align_loss_epi

        # ---- reverse attention branch_2 ----

        qry_fts = layer2_fts[1:2]
        mask_prob_detached = qry_pred_2.detach()
        bd_mask, _ = self.make_boundary_mask(mask_prob_detached)
        N, C, H, W = qry_fts.shape
        xf = self.F_TA(qry_fts, qry_pred_2)
        xb = self.B_TA(qry_fts, 1 - qry_pred_2)
        x = self.TA(qry_fts, bd_mask)
        x = torch.cat((xb, xf, x), 1)
        x = x.view(N, 3 * C, H, W)
        qry_fts2 = self.Fuse(x)

        #################bijiao
        _, _, h, w = qry_fts2.shape
        b_expanded = fg_prototypes_2b[0].unsqueeze(-1).unsqueeze(-1)  # [1, 256, 1, 1]
        b_expanded = b_expanded.expand(-1, -1, h, w)  # [1, 256, h, w]
        f_expanded = fg_prototypes_2[0].unsqueeze(-1).unsqueeze(-1)  # [1, 256, 1, 1]
        f_expanded = f_expanded.expand(-1, -1, h, w)  # [1, 256, h, w]
        combined = torch.cat([qry_fts2, b_expanded, f_expanded], dim=1)  # [1, 768, h, w]
        x = self.conv1(combined)
        x = self.conv2(x)
        qry_pred_3 = self.sigmoid(x)
        #################bijiao

        qry_pred_3 = qry_pred_3 + qry_pred_2

        qry_pred_up = F.interpolate(qry_pred_3, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        if train:
            align_loss_epi = self.alignLoss_bijiao(layer2_fts[0:1].unsqueeze(0), qry_fts2, preds, supp_mask[0])
            align_loss = align_loss + 0.2 * align_loss_epi
        # ---- reverse attention branch_1 ----

        qry_fts = layer1_fts[1:2]
        mask_prob_detached = qry_pred_3.detach()
        bd_mask, _ = self.make_boundary_mask(mask_prob_detached)
        N, C, H, W = qry_fts.shape
        xf = self.F_TA(qry_fts, qry_pred_3)
        xb = self.B_TA(qry_fts, 1 - qry_pred_3)
        x = self.TA(qry_fts, bd_mask)
        x = torch.cat((xb, xf, x), 1)
        x = x.view(N, 3 * C, H, W)
        qry_fts1 = self.Fuse(x)

        #################bijiao
        _, _, h, w = qry_fts1.shape
        b_expanded = fg_prototypes_1b[0].unsqueeze(-1).unsqueeze(-1)  # [1, 256, 1, 1]
        b_expanded = b_expanded.expand(-1, -1, h, w)  # [1, 256, h, w]
        f_expanded = fg_prototypes_1[0].unsqueeze(-1).unsqueeze(-1)  # [1, 256, 1, 1]
        f_expanded = f_expanded.expand(-1, -1, h, w)  # [1, 256, h, w]
        combined = torch.cat([qry_fts1, b_expanded, f_expanded], dim=1)  # [1, 768, h, w]
        x = self.conv1(combined)
        x = self.conv2(x)
        qry_pred_4 = self.sigmoid(x)
        #################bijiao

        qry_pred_4 = qry_pred_4 + qry_pred_3

        qry_pred_up = F.interpolate(qry_pred_4, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        if train:
            align_loss_epi = self.alignLoss_bijiao(layer1_fts[0:1].unsqueeze(0), qry_fts1, preds, supp_mask[0])
            align_loss = align_loss + 0.1 * align_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        if train:
            return output, qry_fts_out, align_loss / supp_bs
        else:
            return output[0].unsqueeze(0), qry_fts_out, align_loss / supp_bs

    def make_boundary_mask(self, prob_mask, kernel_size=5, iters=1, use_soft=True):
        """
        prob_mask: (B,1,H,W) 或 (B,H,W)，值域[0,1] 的前景概率
        返回:
            boundary: (B,1,H,W) 近似边界(0~1)
            interior: (B,1,H,W) 腐蚀后的内区(0~1)
        说明:
            - 膨胀 ~ max_pool2d
            - 腐蚀 ~ 1 - max_pool2d(1 - x)
            - 默认 kernel_size=5（奇数更稳定保持尺寸）
            - 若传入偶数核，会自动中心裁剪保持尺寸一致
        """
        x = prob_mask
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.clamp(0.0, 1.0)

        if not use_soft:
            x = (x > 0.3).float()

        B, C, H, W = x.shape
        k = int(kernel_size)
        pad = (k - 1) // 2 if (k % 2 == 1) else (k // 2)  # 偶数核时将会多 1 像素

        def _center_crop_to(y, H, W):
            h, w = y.shape[-2], y.shape[-1]
            if h == H and w == W:
                return y
            dh, dw = h - H, w - W
            top = max(dh // 2, 0)
            left = max(dw // 2, 0)
            return y[..., top:top + H, left:left + W]

        def _dilate(z):
            y = z
            for _ in range(iters):
                y = F.max_pool2d(y, kernel_size=k, stride=1, padding=pad)
                # 偶数核会 H/W+1，裁回去
                if k % 2 == 0:
                    y = _center_crop_to(y, H, W)
            return y

        def _erode(z):
            # erode(x) = 1 - dilate(1 - x)
            return 1.0 - _dilate(1.0 - z)

        dil = _dilate(x)  # 膨胀
        ero = _erode(x)  # 腐蚀
        boundary = (dil - ero).clamp(0.0, 1.0)
        interior = ero
        return boundary, interior

    def getPred(self, fts, prototype, thresh): # torch.Size([1, 256, 64, 64]) # torch.Size([1, 256])
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W  torch.Size([1, 256, 64, 64])
            prototype: prototype of one semantic class
                expect shape: 1 x C  torch.Size([1, 256])
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):  # torch.Size([1, 256, 64, 64])  torch.Size([1, 256, 256])
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):  # list[list[torch.Size([1, 256])]]
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask): # torch.Size([1, 1, 256, 64, 64])  torch.Size([1, 256, 64, 64])  torch.Size([1, 2, 256, 256])  torch.Size([1, 1, 256, 256])
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # torch.Size([1, 256, 256])
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]  # []
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W' torch.Size([2, 1, 256, 256])

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]  # list[torch.Size([1, 256])]
                fg_prototypes = self.getPrototype([qry_fts_])  # list[torch.Size([1, 256])]

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way],
                                         self.thresh_pred[way])  # N x Wa x H' x W' torch.Size([1, 64, 64])
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                          align_corners=True)  # torch.Size([1, 1, 256, 256])

                # Combine predictions of different feature maps
                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)  # torch.Size([256, 256])
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def alignLoss_bijiao(self, supp_fts, qry_fts, pred, fore_mask):  # torch.Size([1, 1, 256, 64, 64]) torch.Size([1, 256, 64, 64])  torch.Size([1, 2, 256, 256]) torch.Size([1, 1, 256, 256])
        """
        与 alignLoss 类似，但使用 bijiao 风格拼接原型卷积进行分割预测
        （从 query 得到前/背景原型，用来拼接 support 特征做分割）
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # 1. 从 query 预测结果生成伪标签
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # [N, H, W]
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # [(1+Wa), N, H, W]

        loss = torch.zeros(1).to(self.device)

        for way in range(n_ways):
            if way in skip_ways:
                continue

            # 2. 从 query 特征提取前景、背景原型
            qry_fg_fts = [self.getFeatures(qry_fts, pred_mask[way + 1])]
            fg_proto = self.getPrototype([qry_fg_fts])[0]          # torch.Size([1, 256])
            qry_bg_fts = [self.getFeatures(qry_fts, 1 - pred_mask[way + 1])]
            bg_proto = self.getPrototype([qry_bg_fts])[0]         # [1, C]

            for shot in range(n_shots):
                # 3. 拼接 support 特征 + 前/背景原型 → 卷积预测
                _, _, h, w = supp_fts[way, [shot]].shape
                b_expanded = bg_proto.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                f_expanded = fg_proto.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                combined = torch.cat([supp_fts[way, [shot]], b_expanded, f_expanded], dim=1)  # [1,768,H,W]

                x = self.conv1(combined)
                x = self.conv2(x)
                supp_pred = self.sigmoid(x)  # [1,1,H,W]
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # 4. 拼成两通道概率图
                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

                # 5. 构造 support 标签
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # 6. 计算 NLL Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

