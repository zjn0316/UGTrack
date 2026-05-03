import torch
import torch.nn.functional as F

from lib.utils import TensorDict
from lib.train.data.processing import BaseProcessing, stack_tensors
import lib.train.data.processing_utils as prutils
import lib.train.data.uwb_processing_utils as uwb_prutils
import lib.train.data.uwb_transforms as uwb_tfm


class UWBProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        鍙傛暟:
            data - 杈撳叆鏁版嵁锛岄€氬父搴斿寘鍚互涓嬪瓧娈碉細
                   'template_images'銆?search_images'銆?template_anno'銆?search_anno'銆?
                   'search_uwb_seq'銆?search_uwb_gt'銆?search_uwb_conf'銆?
        杩斿洖:
            TensorDict - 澶勭悊鍚庣殑鏁版嵁瀛楀吀銆?
        """
        # 鑻ョ己灏?UWB 瀛楁锛屽垯鐩存帴鏍囪涓烘棤鏁?
        if not all(k in data for k in ['search_uwb_seq', 'search_uwb_gt', 'search_uwb_conf']):
            data['valid'] = False
            return data


        data['search_uwb_conf'] = uwb_prutils.format_search_uwb_conf(data['search_uwb_conf'])

        # 搴旂敤鑱斿悎鍙樻崲
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'], \
                data['search_uwb_seq'], data['search_uwb_gt'] = uwb_tfm.apply_transform_with_uwb(
                    self.transform['joint'],
                    image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'],
                    uwb_seq=data['search_uwb_seq'], uwb_gt=data['search_uwb_gt'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num template/search frames must be 1"

            # 鎵板姩bbox
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 閬垮厤bbox杩囧皬
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                return data

            if s == 'search':
                # 浠ユ壈鍔ㄥ悗鐨勭洰鏍囦负涓績瑁佸壀鍖哄煙锛岀敓鎴愭敞鎰忓姏鎺╃爜锛?
                # 骞跺悓姝ュ皢 UWB 搴忓垪涓?UWB 鏍囨敞鏄犲皠鍒拌鍓悗鍧愭爣绯?
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
                    data[s + '_images'], jittered_anno, data[s + '_anno'], self.search_area_factor[s],
                    self.output_sz[s], masks=data[s + '_masks'])
                data['search_uwb_seq'], data['search_uwb_gt'] = uwb_prutils.jittered_center_crop_uwb(
                    data['search_uwb_seq'], data['search_uwb_gt'], jittered_anno,
                    self.search_area_factor[s], self.output_sz[s])
                # 搴旂敤鐙珛transform
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'], \
                    data['search_uwb_seq'], data['search_uwb_gt'] = uwb_tfm.apply_transform_with_uwb(
                        self.transform[s],
                        image=crops, bbox=boxes, att=att_mask, mask=mask_crops,
                        uwb_seq=data['search_uwb_seq'], uwb_gt=data['search_uwb_gt'], joint=False)
            else:
                # 浠ユ壈鍔ㄥ悗鐨勭洰鏍囦负涓績瑁佸壀鍖哄煙锛岀敓鎴愭敞鎰忓姏鎺╃爜
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
                    data[s + '_images'], jittered_anno, data[s + '_anno'], self.search_area_factor[s],
                    self.output_sz[s], masks=data[s + '_masks'])
                # 搴旂敤鐙珛transform
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 妫€鏌ユ敞鎰忓姏鎺╃爜鏄惁鍏ㄤ负 1锛堟棤鏁堬級
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    return data

                # 妫€鏌ヤ笅閲囨牱鍚庢槸鍚﹀叏涓?1锛堟棤鏁堬級
                feat_size = self.output_sz[s] // 16
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    return data

        data['valid'] = True
        # 鑻ヤ娇鐢?copy-and-paste 澧炲己浣嗘湭鎻愪緵 mask锛屽垯琛ラ浂 mask
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))

        # 鏁寸悊杈撳嚭寮犻噺褰㈢姸
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
