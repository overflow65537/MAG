from typing import Dict, List, Optional, Set
import json
import os

from maa.context import Context


class DataManager:
    def __init__(self, data_path: str):
        self.data_path = data_path

        self.dishes = self._load_dishes()
        self.ingredients = self._load_ingredients()
        self.player_status = self._load_player_status()

        self.shop_info_wrought = False

    def _load_dishes(self) -> Dict[str, List[Dict[str, str | int | Dict[str, int]]]]:
        with open(os.path.join(self.data_path, 'dishes.json'), "r", encoding="UTF-8") as dishes:
            return json.load(dishes)

    def _load_ingredients(self) -> Dict[str, Dict[str, int]]:
        with open(os.path.join(self.data_path, 'ingredients.json'), "r", encoding="UTF-8") as ingredients:
            return json.load(ingredients)

    def _load_player_status(self) -> Dict[str, int | Dict[str, int]]:
        with open(os.path.join(self.data_path, 'player_status.json'), "r", encoding="UTF-8") as player_status:
            return json.load(player_status)

    def update_warehouse_stock(self, warehouse_stock: Dict[str, int]):
        for item_name, new_stock in warehouse_stock.items():
            if item_name in self.player_status['warehouse_stock'].keys():
                self.player_status['warehouse_stock'][item_name] = new_stock

        with open(os.path.join(self.data_path, 'player_status.json'), "w", encoding="UTF-8") as player_status_file:
            player_status_file.write(
                json.dumps(self.player_status, ensure_ascii=False, indent=4)
            )

    def update_merchandise_data(self, merchandise_data: Dict[str, Dict[str, int]]):
        for name, data in merchandise_data.items():
            if "shop_daily_limit" in data.keys():
                self.ingredients[name]["shop_daily_limit"] = data["shop_daily_limit"]
            if "shop_price" in data.keys():
                self.ingredients[name]["shop_price"] = data["shop_price"]
        with open(os.path.join(self.data_path, 'ingredients.json'), "w", encoding="UTF-8") as ingredients:
            ingredients.write(
                json.dumps(self.ingredients, ensure_ascii=False, indent=4)
            )

    def write_shop_today(self, shop_today: Dict[str, Dict[str, int]]):
        with open(os.path.join(self.data_path, 'shop_today.json'), "w", encoding="UTF-8") as shop_today_file:
            shop_today_file.write(
                json.dumps(shop_today, ensure_ascii=False, indent=4)
            )
        self.shop_info_wrought = True

    def load_shop_today(self) -> Dict[str, Dict[str, int]]:
        with open(os.path.join(self.data_path, 'shop_today.json'), "r", encoding="UTF-8") as shop_today_file:
            return json.load(shop_today_file)

    def clean_shop_today(self) -> bool:
        with open(os.path.join(self.data_path, 'shop_today.json'), "w", encoding="UTF-8") as shop_today_file:
            json.dump({}, shop_today_file)

        if self.load_shop_today() == {}:
            return True
        return False


class RestaurantOptimizer(DataManager):
    def __init__(self, data_path: str, context: Context):
        super().__init__(data_path)
        self.context = context
        self._total_available_ingredients: Dict[str, int] = {}
        self._eligible_dishes: Optional[List[Dict[str, str | int | float | Dict[str, int]]]] = None
        self.production_plan: Optional[List[Dict[str, str | int | float]]] = None
        self.demanding_ingredients: Optional[List[str]] = None

    def __del__(self):
        self.clean_shop_today()

    @property
    def result(self):
        return self.production_plan, self.demanding_ingredients

    def run_optimization(self):
        """
        进行决策优化，返回制作菜品的列表和需要购买材料的列表，必须在运行ShopScan动作或使用write_shop_today()后运行
        Returns: production_plan：生产计划，[{"id": 菜品名, "quantity": 制作数量, "bar_ratio": 拖动上架数量条的比例}]；
            demanding_ingredients：所需的可购买食材，['食材名']
        """
        if not self.shop_info_wrought:
            return None

        # 计算可用资源
        shop_today = self.load_shop_today()
        for name, data in self.ingredients.items():
            try:
                self._total_available_ingredients[name] = (
                        shop_today[name]["shop_daily_limit"] + self.player_status["warehouse_stock"].get(name, 0)
                )
            except KeyError:
                self._total_available_ingredients[name] = self.player_status["warehouse_stock"].get(name, 0)

        # 筛选并计算菜品指标
        eligible_dishes: List[Dict[str, str | int | float | Dict[str, int]]] = []
        for cooker, dishes_info in self.dishes.items():
            for dish_info in dishes_info:
                if dish_info["unlock_level"] <= self.player_status["level"][cooker]:
                    dish_data: Dict[str, str | int | float | Dict[str, int]] = dish_info.copy()
                    dish_data["profit_rate"] = dish_info["profit"] / dish_info["sell_time"]
                    eligible_dishes.append(dish_data)
        eligible_dishes.sort(key=lambda dish: dish["profit_rate"], reverse=True)
        self._eligible_dishes = eligible_dishes

        self.production_plan = self._decision_optimize()
        self.demanding_ingredients = self._buying_demand(shop_today)
        return self.result

    def _decision_optimize(self) -> Optional[List[Dict[str, str | int | float]]]:
        """
        进行决策优化，寻找收益最大化的菜品种类及其制作数量
        仅支持在run_optimization()中运行
        :return: 菜品列表，元素为字典：{"id": 菜品名称, "quantity": 制作数量}
        """
        assert self._eligible_dishes is not None

        production_plan: List[Dict[str, str | int | float]] = []
        total_time = 1440
        remaining_ingredients = self._total_available_ingredients.copy()

        # 计算最大可制作份数
        for dish in self._eligible_dishes:
            if len(production_plan) >= self.player_status["menu_slots"]:
                break

            # 基于食材上限
            max_from_ingredients = float('inf')
            can_make_flag = True
            for name, required_amount in dish["ingredients"].items():
                if remaining_ingredients.get(name, 0) == 0:
                    can_make_flag = False
                    break
                can_make = remaining_ingredients.get(name, 0) // required_amount
                max_from_ingredients = min(max_from_ingredients, can_make)
            if not can_make_flag or max_from_ingredients == 0:
                continue
            # 基于时间上限
            max_from_time = total_time // dish["sell_time"]

            # 如果可以制作，则添加到制作计划并更新剩余食材
            quantity_to_make = min(max_from_ingredients, max_from_time)
            if quantity_to_make > 0:
                bar_ratio = quantity_to_make / max_from_ingredients
                production_plan.append({
                    "id": dish["dish_id"],
                    "quantity": quantity_to_make,
                    "bar_ratio": bar_ratio
                })
                for name, required_amount in dish["ingredients"].items():
                    remaining_ingredients[name] -= required_amount * quantity_to_make

        if not production_plan:
            return None
        return production_plan

    def _buying_demand(self, shop_today: Dict[str, Dict[str, int]]) -> Optional[List[str]]:
        """
        考虑到可持续性运营，决策得出的菜品所消耗的、可在商店中补货的原料都需要满额补货
        需要在_decision_optimize()后运行
        :return: 需要购买的原料列表
        """
        if self.production_plan is None:
            return None

        required_ingredients: Set[str] = set()
        # 获取在决策列表中的菜品所消耗的原料
        dishes_requirements: Dict[str, Dict[str, int]] = {
            dish["dish_id"]: dish["ingredients"] for dish in self._eligible_dishes
            if dish["dish_id"] in [plan_dish["id"] for plan_dish in self.production_plan]
        }

        # 遍历决策菜品列表，若其原料可购买则加入结果
        for ingredients in dishes_requirements.values():
            for ingredient in ingredients.keys():
                if ingredient in shop_today.keys():
                    required_ingredients.add(ingredient)

        return list(required_ingredients)
