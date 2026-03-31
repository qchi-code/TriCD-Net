import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Res101Encoder
from .decoder_p import FEBR
from .deconder import EnhancedDecoder, UCI


class FewShotSeg(nn.Module):
    """Few-shot segmentation network with FEBR, DDCS, CDCF, and UCI."""

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()
        self.encoder = Res101Encoder(
            replace_stride_with_dilation=[True, True, False],
            pretrained_weights=pretrained_weights,
        )
        self.scaler = 20.0
        self.device = torch.device('cuda')
        self.criterion = nn.NLLLoss()

        self.decoder1 = EnhancedDecoder()

        self.febr1 = FEBR(dim=256, num_heads=8, bias=False, LayerNorm_type='WithBias')
        self.febr2 = FEBR(dim=256, num_heads=8, bias=False, LayerNorm_type='WithBias')
        self.febr3 = FEBR(dim=256, num_heads=8, bias=False, LayerNorm_type='WithBias')

        self.uci1 = UCI(seg_channels=256, rec_channels=256, num_heads=8)
        self.uci2 = UCI(seg_channels=256, rec_channels=512, num_heads=8)
        self.uci3 = UCI(seg_channels=256, rec_channels=1024, num_heads=8)

        self.reduceconv4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)
        self.reduceconv3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.reduceconv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.reduceconv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)

        self.conv1 = nn.Conv2d(768, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """Run episodic training or inference for one-way few-shot segmentation."""
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = 1
        assert self.n_ways == 1
        assert self.n_queries == 1

        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask], dim=0)
        supp_mask = supp_mask.view(supp_bs, self.n_ways, self.n_shots, *img_size)

        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(qry_imgs, dim=0)], dim=0
        )

        img_fts, tao = self.encoder(imgs_concat[0:2])
        self.t = tao[1:2]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        qry_fts_out = None
        rec_fts = None
        if train:
            rec_fts, _ = self.encoder(imgs_concat[2:3])
            rec_inputs = [rec_fts["layer1"], rec_fts["layer2"], rec_fts["layer3"], rec_fts["layer4"]]
            qry_fts_out = self.decoder1(rec_inputs)

        outputs = []
        align_loss = torch.zeros(1, device=imgs_concat.device)

        layer3_fts = self.reduceconv3(img_fts["layer3"])
        layer2_fts = self.reduceconv2(img_fts["layer2"])
        layer1_fts = self.reduceconv1(img_fts["layer1"])

        supp_fts_3 = self.getFeatures(layer3_fts[0:1], supp_mask[0][0])
        fg_prototypes_3 = self.getPrototype([[supp_fts_3]])
        supp_fts_3b = self.getSoftBgFeatures(layer3_fts[0:1], supp_mask[0][0], fg_prototypes_3[0])
        bg_prototypes_3 = self.getPrototype([[supp_fts_3b]])

        supp_fts_2 = self.getFeatures(layer2_fts[0:1], supp_mask[0][0])
        fg_prototypes_2 = self.getPrototype([[supp_fts_2]])
        supp_fts_2b = self.getSoftBgFeatures(layer2_fts[0:1], supp_mask[0][0], fg_prototypes_2[0])
        bg_prototypes_2 = self.getPrototype([[supp_fts_2b]])

        supp_fts_1 = self.getFeatures(layer1_fts[0:1], supp_mask[0][0])
        fg_prototypes_1 = self.getPrototype([[supp_fts_1]])
        supp_fts_1b = self.getSoftBgFeatures(layer1_fts[0:1], supp_mask[0][0], fg_prototypes_1[0])
        bg_prototypes_1 = self.getPrototype([[supp_fts_1b]])

        qry_pred_1 = torch.stack(
            [self.getPred(layer3_fts[1:2], fg_prototypes_3[0], self.thresh_pred[way]) for way in range(self.n_ways)],
            dim=1,
        )

        qry_pred_up = F.interpolate(qry_pred_1, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        if train:
            align_loss_epi = self.alignLoss(layer3_fts[0:1].unsqueeze(0), layer3_fts[1:2], preds, supp_mask[0])
            align_loss += align_loss_epi

        qry_fts = layer3_fts[1:2]
        if train and rec_fts is not None:
            qry_fts = self.uci3(qry_fts, rec_fts["layer3"], qry_pred_1.detach())
        mask_prob_detached = qry_pred_1.detach()
        bd_mask, _ = self.make_boundary_mask(mask_prob_detached)
        qry_fts3 = self.febr3(qry_fts, mask_prob_detached, bd_mask)

        _, _, h, w = qry_fts3.shape
        b_expanded = bg_prototypes_3[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        f_expanded = fg_prototypes_3[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        combined = torch.cat([qry_fts3, b_expanded, f_expanded], dim=1)
        x = self.conv1(combined)
        x = self.conv2(x)
        qry_pred_2 = self.sigmoid(x) + qry_pred_1

        qry_pred_up = F.interpolate(qry_pred_2, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        qry_fts = layer2_fts[1:2]
        if train and rec_fts is not None:
            qry_fts = self.uci2(qry_fts, rec_fts["layer2"], qry_pred_2.detach())
        mask_prob_detached = qry_pred_2.detach()
        bd_mask, _ = self.make_boundary_mask(mask_prob_detached)
        qry_fts2 = self.febr2(qry_fts, mask_prob_detached, bd_mask)

        _, _, h, w = qry_fts2.shape
        b_expanded = bg_prototypes_2[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        f_expanded = fg_prototypes_2[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        combined = torch.cat([qry_fts2, b_expanded, f_expanded], dim=1)
        x = self.conv1(combined)
        x = self.conv2(x)
        qry_pred_3 = self.sigmoid(x) + qry_pred_2

        qry_pred_up = F.interpolate(qry_pred_3, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        qry_fts = layer1_fts[1:2]
        if train and rec_fts is not None:
            qry_fts = self.uci1(qry_fts, rec_fts["layer1"], qry_pred_3.detach())
        mask_prob_detached = qry_pred_3.detach()
        bd_mask, _ = self.make_boundary_mask(mask_prob_detached)
        qry_fts1 = self.febr1(qry_fts, mask_prob_detached, bd_mask)

        _, _, h, w = qry_fts1.shape
        b_expanded = bg_prototypes_1[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        f_expanded = fg_prototypes_1[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        combined = torch.cat([qry_fts1, b_expanded, f_expanded], dim=1)
        x = self.conv1(combined)
        x = self.conv2(x)
        qry_pred_4 = self.sigmoid(x) + qry_pred_3

        qry_pred_up = F.interpolate(qry_pred_4, size=img_size, mode='bilinear', align_corners=True)
        preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
        outputs.append(preds)

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        if train:
            return output, qry_fts_out, align_loss / supp_bs
        return output[0].unsqueeze(0), qry_fts_out, align_loss / supp_bs

    def make_boundary_mask(self, prob_mask, kernel_size=5, iters=1, use_soft=True):
        """Generate soft boundary and interior masks from the foreground probability map."""
        x = prob_mask
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.clamp(0.0, 1.0)

        if not use_soft:
            x = (x > 0.3).float()

        _, _, h, w = x.shape
        k = int(kernel_size)
        pad = (k - 1) // 2 if (k % 2 == 1) else (k // 2)

        def _center_crop_to(y, hh, ww):
            cur_h, cur_w = y.shape[-2], y.shape[-1]
            if cur_h == hh and cur_w == ww:
                return y
            dh, dw = cur_h - hh, cur_w - ww
            top = max(dh // 2, 0)
            left = max(dw // 2, 0)
            return y[..., top:top + hh, left:left + ww]

        def _dilate(z):
            y = z
            for _ in range(iters):
                y = F.max_pool2d(y, kernel_size=k, stride=1, padding=pad)
                if k % 2 == 0:
                    y = _center_crop_to(y, h, w)
            return y

        def _erode(z):
            return 1.0 - _dilate(1.0 - z)

        dil = _dilate(x)
        ero = _erode(x)
        boundary = (dil - ero).clamp(0.0, 1.0)
        interior = ero
        return boundary, interior

    def getPred(self, fts, prototype, thresh):
        """Compute the coarse foreground probability from feature-prototype similarity."""
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred

    def getFeatures(self, fts, mask):
        """Extract the masked average pooled prototype for one region."""
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        return masked_fts

    def getSoftBgFeatures(self, fts, fg_mask, fg_proto):
        """Extract the Soft-MAP background prototype using foreground-feature similarity weighting."""
        if fg_mask.dim() == 3:
            fg_mask = fg_mask.unsqueeze(1)
        fts = F.interpolate(fts, size=fg_mask.shape[-2:], mode='bilinear', align_corners=True)
        cos_sim = F.cosine_similarity(fts, fg_proto[..., None, None], dim=1).unsqueeze(1)
        weights = (1.0 - cos_sim) * (1.0 - fg_mask)
        bg_fts = torch.sum(fts * weights, dim=(-2, -1)) / (weights.sum(dim=(-2, -1)) + 1e-5)
        return bg_fts

    def getPrototype(self, fg_fts):
        """Average shot-level prototypes to obtain the class prototype."""
        n_shots = len(fg_fts[0])
        fg_prototypes = [
            torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in fg_fts
        ]
        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """Compute the PANet-style feature alignment loss."""
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        loss = torch.zeros(1, device=qry_fts.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways
        return loss
