import torch
from .base_actor import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.ce_utils import adjust_keep_rate, generate_mask_cond
from lib.utils.heapmap_utils import generate_heatmap


class UGTrackActor(BaseActor):
    """用于训练 UGTrack 模型的 Actor 类，负责执行前向传播并计算损失。"""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg
        self.stage = int(cfg.TRAIN.STAGE) if cfg is not None else 1  # [UGTrack独有] 双阶段训练开关
        self.iter_count = 0

    def __call__(self, data):
        """
        参数:
            data - 输入数据字典，包含 UWB 序列、图像帧及标注信息。
        返回:
            loss   - 训练的总损失（用于反向传播）
            status - 包含详细损失项和指标的字典（用于日志记录）
        """
        self.iter_count += 1

        # 执行前向传播获取预测输出
        out_dict = self.forward_pass(data)

        # 计算多项损失并汇总
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        """执行模型的前向传播逻辑。"""
        # [UGTrack独有] 阶段1：仅 UWB 分支前向
        if self.stage == 1:
            search_uwb_seq = data["search_uwb_seq"].squeeze(0).float()
            return self.net(search_uwb_seq=search_uwb_seq, stage=1)

        # [UGTrack独有] 阶段2：处理 UWB 序列输入
        search_uwb_seq = data['search_uwb_seq'].squeeze(0).float()

        # 目前仅支持 1 个模板帧和 1 个搜索区域
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        # 处理模板图像列表
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        # 处理搜索区域图像
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        # 如果启用候选消融 (Candidate Elimination, CE)，生成对应的掩码和保留率
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            ce_keep_rate = adjust_keep_rate(data['epoch'],
                                            warmup_epochs=self.cfg.TRAIN.CE_START_EPOCH,
                                            total_epochs=self.cfg.TRAIN.CE_START_EPOCH + self.cfg.TRAIN.CE_WARM_EPOCH,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        # 调用核心神经网络进行推理
        out_dict = self.net(template=template_list,
                            search=search_img,
                            search_uwb_seq=search_uwb_seq,# [UGTrack独有] 阶段2注入 UWB 序列
                            stage=2,# [UGTrack独有] 显式指定联合训练阶段
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """计算预测结果与真实标签之间的各项损失。"""
        # [UGTrack独有] 阶段1：UWB 双分支损失
        if self.stage == 1:
            # 提取真值
            search_uwb_gt = gt_dict["search_uwb_gt"].squeeze(0).float()
            search_uwb_conf = gt_dict["search_uwb_conf"].squeeze(0).float()

            # 计算 UWB 损失 (回归 + 置信度)
            pred_loss = self.objective['uwb_pred'](pred_dict["pred_uv"], search_uwb_gt[..., :2])
            conf_loss = self.objective['uwb_conf'](pred_dict["uwb_conf_logit"], search_uwb_conf)

            loss = self.loss_weight["uwb_pred"] * pred_loss + self.loss_weight["uwb_conf"] * conf_loss

            # 记录详细的状态字典，用于日志打印（如 Wandb/Tensorboard）
            status = {
                "Loss/uwb_total": loss.item(),
                "Loss/uwb_pred": pred_loss.item(),
                "Loss/uwb_conf": conf_loss.item(),
            }
            return loss, status

        # 提取真实边界框 (Search Annotation)
        gt_bbox = gt_dict["search_anno"][-1]
        # 生成真实的热力图 (Gaussian Map)
        gt_gaussian_maps = generate_heatmap(
            gt_dict["search_anno"],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE,
        )
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # 获取模型预测的边界框
        pred_boxes = pred_dict["pred_boxes"]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network output contains NaN")

        # 将预测框和真实框统一格式以便计算 (cxcywh -> xyxy)
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)
        gt_boxes_vec = gt_boxes_vec.clamp(min=0.0, max=1.0)

        # 1. 计算 GIoU 和 IoU
        try:
            giou_loss, iou = self.objective["giou"](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
            iou = torch.tensor(0.0, device=pred_boxes.device)

        # 2. 计算 L1 损失 (坐标偏移误差)
        l1_loss = self.objective["l1"](pred_boxes_vec, gt_boxes_vec)
        # 3. 计算 Focal Loss (位置分类误差)
        if "score_map" in pred_dict:
            location_loss = self.objective["focal"](pred_dict["score_map"], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # 按照配置文件中的权重进行多任务损失汇总
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        
        if return_status:
            # 记录详细的状态字典，用于日志打印（如 Wandb/Tensorboard）
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "IoU": mean_iou.item(),
            }
            # [UGTrack独有] 记录 UWB 分支统计信息
            if "uwb_conf_pred" in pred_dict:
                status["UWB/conf_mean"] = pred_dict["uwb_conf_pred"].detach().mean().item()
            return loss, status
        else:
            return loss
