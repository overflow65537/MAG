from typing import Dict, List, Optional, Tuple, Set

from maa.define import Rect
from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from pypinyin import lazy_pinyin
import torch
import json
import cv2
import os
import numpy as np

from maa.context import Context
from maa.custom_recognition import CustomRecognition


# 先定义一个功能类：文本后处理，便于后期全局变量调用
class DictPostprocess:
    def __init__(self, characters_data_path: str):
        self.names, self.confusions = self._load_character_data(characters_data_path)
        self.confusions: Dict[str, Set[str]] = {primal: set(similar) for primal, similar in self.confusions.items()}
        self.allowed_chars = set(ch for name in self.names for ch in name)

    def normalize_to_whitelist(self, text: str) -> str:
        """
        主要方法1：将OCR结果清洗到白名单字符集：
        - 非白名单字符尝试按混淆表替换到最近的白名单字符
        - 若无替换，直接删除该字符
        """
        norm_chars = []
        for ch in text:
            if ch in self.allowed_chars:
                norm_chars.append(ch)
                continue
            # 尝试用混淆表映射到白名单
            candidates = set()
            if ch in self.confusions:
                candidates |= (self.confusions[ch] & self.allowed_chars)
            # 反向：哪些白名单字符把 ch 当作混淆
            for k, vs in self.confusions.items():
                if ch in vs and k in self.allowed_chars:
                    candidates.add(k)
            if candidates:
                # 选一个（这里任意挑第一个；也可以按字频或编辑距离进一步挑）
                norm_chars.append(list(candidates)[0])
            # 否则丢弃
        return ''.join(norm_chars)

    def pick_best_name(self, ocr_text: str) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """
        主要方法2：获取最佳匹配
        先 bigram 收缩候选，再算分选Top-1。
        对二字名使用更高阈值。
        返回：best_name, best_score, top3列表
        """
        candidates = self._prefilter_candidates_by_bigram(ocr_text)
        ranked = sorted(((nm, self._score_name(ocr_text, nm)) for nm in candidates),
                        key=lambda x: x[1], reverse=True)
        if not ranked:
            return None, 0.0, []
        top3 = ranked[:3]
        best, sc = ranked[0]
        thr = 0.86 if len(best) == 2 else 0.75
        if sc >= thr:
            return best, sc, top3
        return None, sc, top3

    @staticmethod
    def _load_character_data(data_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
        with open(data_path, "r", encoding="UTF-8") as file:
            data = json.load(file)
        return data["names"], data["confusions"]

    # ========== 词典后处理（加权编辑距离 + 拼音 + 子串加分） ==========
    def _sub_cost(self, major: str, comparison: str) -> float:
        """计算字符替换的加权成本"""
        if major == comparison:  # 完全相同
            return 0
        try:  # 同音
            if lazy_pinyin(major) == lazy_pinyin(comparison):
                return 0.4
        except Exception:
            pass
        # 形近/常混
        if major in self.confusions.keys() and comparison in self.confusions[major]:
            return 0.3
        if comparison in self.confusions.keys() and major in self.confusions[comparison]:
            return 0.3
        return 1

    def _weighted_levenshtein(self, major: str, comparison: str) -> float:
        """计算加权编辑距离相似度"""
        m, n = len(major), len(comparison)
        dp = np.zeros((m + 1, n + 1), dtype=np.float64)
        # 初始化第一列和第一行
        dp[1:, 0] = np.arange(1, m + 1, dtype=np.float64)
        dp[0, 1:] = np.arange(1, n + 1, dtype=np.float64)
        # 动态规划计算
        for row in range(1, m + 1):
            for col in range(1, n + 1):
                c_sub = dp[row - 1, col - 1].item() + self._sub_cost(major[row - 1], comparison[col - 1])
                c_del = dp[row - 1, col].item() + 1.0
                c_ins = dp[row, col - 1].item() + 1.0
                dp[row, col] = min(c_sub, c_del, c_ins)

        denom = max(m, n) if max(m, n) else 1
        dist: float = dp[m, n].item() / denom
        return 1.0 - dist  # 相似度[0,1]

    @staticmethod
    def _pinyin_sim(major: str, comparison: str) -> float:
        """计算拼音相似度"""
        ps = ''.join(lazy_pinyin(major))
        pt = ''.join(lazy_pinyin(comparison))
        if not ps and not pt:
            return 1.0
        # 使用NumPy优化普通编辑距离计算
        m, n = len(ps), len(pt)
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)
        # 初始化
        dp[:, 0] = np.arange(m + 1)
        dp[0, :] = np.arange(n + 1)
        # 动态规划
        for row in range(1, m + 1):
            for col in range(1, n + 1):
                cost = 0 if ps[row - 1] == pt[col - 1] else 1
                dp[row, col] = min(
                    dp[row - 1, col].item() + 1,
                    dp[row, col - 1].item() + 1,
                    dp[row - 1, col - 1].item() + cost
                )
        denom = max(m, n) if max(m, n) else 1
        return 1.0 - dp[m, n].item() / denom

    @staticmethod
    def _substring_bonus(major: str, comparison: str) -> float:
        """连续子串加分"""
        # 命中任意2字相邻子串加分
        if len(major) < 2 or len(comparison) < 2:
            return 0.0
        major_2successive = {major[i:i + 2] for i in range(len(major) - 1)}
        comparison_2successive = {comparison[i:i + 2] for i in range(len(comparison) - 1)}
        inter = major_2successive & comparison_2successive
        return 0.12 if inter else 0.0

    def _score_name(self, ocr_text: str, name: str) -> float:
        """综合评分"""
        w1, w2 = 0.7, 0.25  # 主靠编辑距离，其次拼音
        sim1 = self._weighted_levenshtein(ocr_text, name)
        sim2 = self._pinyin_sim(ocr_text, name)
        bonus = self._substring_bonus(ocr_text, name)
        return min(1.0, w1 * sim1 + w2 * sim2 + bonus)

    def _prefilter_candidates_by_bigram(self, obs: str) -> List[str]:
        """
        收缩候选集
        利用二字子串快速收缩候选集；若未命中，回退到全量。
        """
        if len(obs) < 2:
            return self.names
        grams = {obs[i:i + 2] for i in range(len(obs) - 1)}
        hits = [nm for nm in self.names if any(g in nm for g in grams)]
        # 避免过度收缩，保底留下一些
        return hits if hits else self.names


# ============ 创建后续要用到的全局变量 ============
# 创建全局变量
config_path: Optional[str] = None
postprocessor: Optional[DictPostprocess] = None
up_sampler: Optional[RealESRGANer] = None


class RobustlyNameRecognize(CustomRecognition):
    """
    用于在低分辨率场景下对角色名进行高精准度的识别
    需要在 custom_action_param 中传入 custom_task_config/idol_training 文件夹的路径
    """
    def analyze(self,
                context: Context,
                argv: CustomRecognition.AnalyzeArg
                ) -> CustomRecognition.AnalyzeResult:
        # 首次运行时，加载后处理器和超分辨率模型
        global config_path, postprocessor, up_sampler
        if config_path is None:
            config_path = json.loads(argv.custom_recognition_param)
        if postprocessor is None:
            postprocessor = DictPostprocess(os.path.join(config_path, "character.json"))
        if up_sampler is None:
            try:
                up_sampler = RealESRGANer(
                    scale=4,
                    model_path=os.path.join(config_path, "realesr-animevideov3.pth"),
                    model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4,
                                          act_type='prelu'),
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=torch.cuda.is_available()
                )
            except Exception as e:
                context.run_task("push_message", {
                    "push_message": {
                        "focus": {
                            "start": f"[size:20][color:red]由于{str(e)}，超分辨率模型未加载[/color][/size]",
                            "succeeded": "[size:20][color:red]文字识别可能出错[/color][/size]"
                        }
                    }
                })

        # 对图片进行预处理并OCR
        roi = argv.roi
        img_input, box = self.preprocess_text(self.pic_SR(argv.image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]))
        name_reco_detail = context.run_recognition("character_reco", img_input, {
            "recognition": {
                "type": "OCR",
                "param": {
                    "only_rec": True,
                    "expected": "[\\u4e00-\\u9fff]+"
                }
            }
        })
        if name_reco_detail:
            name_row = name_reco_detail.best_result.text \
                if name_reco_detail.best_result is not None else name_reco_detail.filterd_results[0]
        else:
            return CustomRecognition.AnalyzeResult(None, "")

        # 对得到的结果进行后处理
        name_cleaned = postprocessor.normalize_to_whitelist(name_row)  # 清洗到白名单
        obs = name_cleaned if len(name_cleaned) > 2 else name_row  # 若清洗后过短，尝试直接用raw进行匹配（避免过度删除）
        best_name, score, top3 = postprocessor.pick_best_name(obs)  # 用词典后处理做最终判定
        return CustomRecognition.AnalyzeResult(box=Rect(*box), detail=best_name)

    @staticmethod
    def pic_SR(pic: np.ndarray) -> np.ndarray:
        """使用Real-ESRGAN进行超分辨率，不可用时原样返回"""
        global up_sampler
        if up_sampler:
            output, _ = up_sampler.enhance(pic, outscale=4)
            return output
        else:
            return pic

    @staticmethod
    def preprocess_text(img: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """预处理文字图片，并将得到的二值图按前景区域做紧致裁剪"""
        # Gamma矫正提高白字亮度
        img = img.astype(np.float32) / 255.0
        gamma = 0.85  # <1 提亮，>1 变暗
        img = np.clip(img ** gamma, 0, 1)
        img = (img * 255).astype(np.uint8)

        # 转HSV做“白色”分割（低饱和 + 高明度）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        white_mask1 = (S <= int(0.32 * 255)) & (V >= int(0.70 * 255))

        # 转Lab做“接近灰白”分割（高亮 + a/b接近中性）
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L, A, Bc = cv2.split(lab)
        white_mask2 = (L >= 170) & (np.abs(A.astype(np.int16) - 128) <= 16) & (np.abs(Bc.astype(np.int16) - 128) <= 18)

        white_mask = (white_mask1 | white_mask2).astype(np.uint8) * 255

        # 背景蓝色（近 #657EA2 / #6C85B4），可选排除
        # 这里用“蓝色范围”弱化背景，但主要还是白字阈值在起作用
        # lower_blue = np.array([90, 30, 50])  # HSV 粗范围（需按实际微调）
        # upper_blue = np.array([140, 255, 255])
        # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 白字优先，蓝底不强行排除，避免描边被误伤

        # 形态学连接细笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 自适应阈值融合：用白字mask作为权重引导二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 局部二值
        bin_local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, blockSize=21, C=-5)
        # 合成：白字区域用局部二值，其他区域设为背景
        bin_img = np.zeros_like(gray)
        bin_img[white_mask > 0] = bin_local[white_mask > 0]

        # 进一步清理小连通域噪声
        nb_components, labels, stats, _ = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8),
                                                                           connectivity=8)
        clean = np.zeros_like(bin_img)
        # 根据最小面积阈值过滤（按ROI尺寸自适应）
        h, w = bin_img.shape
        min_area = max(8, (h * w) // 2000)  # 经验阈值
        for i in range(1, nb_components):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean[labels == i] = 255

        # 轻微锐化
        sharped = cv2.addWeighted(
            clean,
            1.5,
            cv2.GaussianBlur(clean, (0, 0), 1.0),
            -0.5,
            0
        )

        # 紧致裁剪
        pad = 4
        ys, xs = np.where(sharped > 0)
        if len(xs) == 0 or len(ys) == 0:
            return sharped
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, sharped.shape[1] - 1)
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, sharped.shape[0] - 1)
        return sharped[y1:y2 + 1, x1:x2 + 1], [x1, y1, x2-x1, y2-y1]
