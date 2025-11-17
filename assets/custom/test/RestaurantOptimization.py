from typing import List, Dict, Tuple, Optional
from itertools import combinations
from dataclasses import dataclass


@dataclass
class Dish:
    """菜品类"""
    name: str
    cookware: str  # 所属厨具
    price: float  # 售卖价格
    time: float  # 预计售卖时间（分钟）
    unlock_level: int  # 解锁等级
    ingredients: Dict[str, int]  # 消耗食材 {食材名: 数量}

    @property
    def profit_rate(self) -> float:
        """收益率（每分钟收益）"""
        return self.price / self.time if self.time > 0 else 0

    def __repr__(self):
        return f"{self.name}(收益率:{self.profit_rate:.2f})"


@dataclass
class MenuSolution:
    """单个菜品的上架方案"""
    dish: Dish  # 上架菜品
    count: int  # 上架数量
    bar_ratio: float  # 需要拖动的进度条比例


class RestaurantDataManager:
    """餐厅相关数据的读取和管理"""