from typing import List, Dict, Tuple, Set, Literal
from itertools import combinations
from dataclasses import dataclass
import os
import json


@dataclass
class Dish:
    """菜品类"""
    name: str
    cookware: Literal["炒锅", "烤箱", "蒸笼", "煮锅"]  # 所属厨具
    price: int  # 售卖价格
    time: int  # 预计售卖时间（分钟）
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
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.shop_info_wrought = False
        self.all_dishes = self._load_dishes()
        self.levels, self.storage = self._load_player_status()

    def _load_dishes(self) -> List[Dish]:
        dishes: List[Dish] = []
        with open(os.path.join(self.data_path, 'dishes.json'), "r", encoding="UTF-8") as file_dishes:
            dic_dishes = json.load(file_dishes)
            for cookware, current_cookware_dishes in dic_dishes.items():
                for dic_dish in current_cookware_dishes:
                    dishes.append(Dish(
                        name=dic_dish['dish_id'],
                        cookware=cookware,
                        price=dic_dish.get('profit', 0),
                        time=dic_dish.get('sell_time', 0),  # 已在Dish类中解决了除零错误
                        unlock_level=dic_dish.get('unlock_level', 3),
                        ingredients=dic_dish['ingredients'],
                    ))
        return dishes

    def _load_player_status(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """返回元组：(厨具等级, 仓库储量)"""
        with open(os.path.join(self.data_path, 'player_status.json'), "r", encoding="UTF-8") as file_player_status:
            player_status = json.load(file_player_status)
        return player_status['level'], player_status['warehouse_stock']

    def update_storage(self, stock: Dict[str, int]):
        with open(os.path.join(self.data_path, 'player_status.json'), "r+", encoding="UTF-8") as player_status_file:
            player_status: Dict[str, int|Dict[str, int]] = json.load(player_status_file)
            player_status['warehouse_stock'] = {name:stock for name, stock in stock.items()}
            player_status_file.seek(0)
            json.dump(player_status, player_status_file, indent=4, ensure_ascii=False)

    def write_shop_info_today(self, shop_today: Dict[str, int]):
        with open(os.path.join(self.data_path, 'shop_today.json'), "w", encoding="UTF-8") as shop_today_file:
            shop_today_file.write(
                json.dumps(shop_today, ensure_ascii=False, indent=4)
            )
        self.shop_info_wrought = True

    def load_shop_info_today(self) -> Dict[str, int]:
        with open(os.path.join(self.data_path, 'shop_today.json'), "r", encoding="UTF-8") as shop_today_file:
            return json.load(shop_today_file)

    def clean_shop_info_today(self) -> bool:
        with open(os.path.join(self.data_path, 'shop_today.json'), "w", encoding="UTF-8") as shop_today_file:
            json.dump({}, shop_today_file)

        if self.load_shop_info_today() == {}:
            return True
        return False


class RestaurantOptimizer(RestaurantDataManager):
    def __init__(self, data_path: str, time_limit: float = 24, max_slots: int = 2):
        """
        :param data_path: 餐厅相关文件的存储路径
        :param time_limit: 时间限制（小时），默认24小时
        """
        super().__init__(data_path)
        self.time_limit = int(time_limit * 60)  # 转换为分钟
        self.max_slots = max_slots
        self.unlocked_dishes = [
            dish for dish in self.all_dishes if dish.unlock_level <= self.levels[dish.cookware]
        ]  # 筛选已解锁的菜品

        # 计算总可用食材（仓库+商店）
        self.total_ingredients = self.storage.copy()
        for ingredient, amount in self.load_shop_info_today().items():
            self.total_ingredients[ingredient] = self.total_ingredients.get(ingredient, 0) + amount

    def _calc_time_limit(self, dish: Dish) -> int:
        """计算单个菜品在时间限制内最多能做多少份"""
        if dish.time <= 0:
            return 0
        return int(self.time_limit / dish.time)

    def _calc_ingredient_limit(self, dish: Dish, available: Dict[str, int]) -> int:
        """计算食材限制下最多能做多少份"""
        max_count = float('inf')
        for ingredient, required in dish.ingredients.items():
            if required > 0:
                available_amount = available.get(ingredient, 0)
                max_count = min(max_count, available_amount // required)
        return int(max_count) if max_count != float('inf') else 0

    def _optimize_single_dish(self, dish: Dish) -> Tuple[int, float]:
        """优化单个菜品的制作数量"""
        count = min(self._calc_time_limit(dish), self._calc_ingredient_limit(dish, self.total_ingredients))
        profit = count * dish.price
        return count, profit

    @staticmethod
    def _calc_bar_ratio(dish: Dish, count: int, available_ingredients: Dict[str, int]) -> Tuple[float, Dict[str, int]]:
        """根据菜品、制作数量及当前食材计算进度条拖动的比例和制作后的剩余食材"""
        ratio: float = 0
        current_ingredients = available_ingredients.copy()

        for ingredient, required_amount in dish.ingredients.items():
            current_amount = current_ingredients.get(ingredient, 0)
            remaining_amount = current_amount - required_amount * count
            if remaining_amount < 0: remaining_amount = 0
            current_ratio = (current_amount - remaining_amount) / current_amount

            # 可制作的数量取决于短板（即剩余最少的食材），其对应的进度条比例最大
            ratio = max(current_ratio, ratio)
            current_ingredients[ingredient] = remaining_amount

        return ratio, current_ingredients

    def _optimize_two_dishes(self, dish1: Dish, dish2: Dish) -> Tuple[List[int], float]:
        """
        优化两个菜品的组合
        由于两个菜品并行售卖，时间限制是独立的，需要在食材约束下最大化总收益
        """
        time_limit1 = self._calc_time_limit(dish1)
        time_limit2 = self._calc_time_limit(dish2)
        best_counts, best_profit = [0, 0], 0

        # 策略：枚举第一个菜品的数量，计算第二个菜品的最大数量
        # 优化：如果菜品1的收益率更高，优先枚举收益率低的
        if dish1.profit_rate < dish2.profit_rate:
            dish1, dish2 = dish2, dish1
            time_limit1, time_limit2 = time_limit2, time_limit1
            swapped = True
        else:
            swapped = False

        # 枚举第一个菜品的数量
        for count1 in range(time_limit1 + 1):
            # 计算已使用的食材
            used_ingredients = {}
            can_make = True

            for ingredient, required in dish1.ingredients.items():
                total_needed = count1 * required
                if total_needed > self.total_ingredients.get(ingredient, 0):
                    can_make = False
                    break
                used_ingredients[ingredient] = total_needed

            if not can_make:
                break

            # 计算剩余食材
            remaining_ingredients = {}
            for ingredient, total in self.total_ingredients.items():
                remaining_ingredients[ingredient] = total - used_ingredients.get(ingredient, 0)

            # 计算菜品2在剩余食材下的最大数量
            ingredient_limit2 = self._calc_ingredient_limit(
                dish2, remaining_ingredients
            )
            count2 = min(time_limit2, ingredient_limit2)

            # 计算总收益
            profit = count1 * dish1.price + count2 * dish2.price

            if profit > best_profit:
                best_profit = profit
                if swapped:
                    best_counts = [count2, count1]
                else:
                    best_counts = [count1, count2]

        if swapped:
            return best_counts, best_profit
        else:
            return best_counts, best_profit

    def find_best_solution(self) -> Tuple[List[MenuSolution], Set[str]]:
        """找到最优菜品组合方案及需要购买的食材"""
        best_solution: Dict[str, List[Dish]|List[int]|int|List[float]] = {
            'dishes': [],
            'counts': [],
            'profit': 0,
            'total_time_hours': [],
        }

        # 尝试单个菜品
        for dish in self.unlocked_dishes:
            count, profit = self._optimize_single_dish(dish)
            if profit > best_solution['profit']:
                best_solution = {
                    'dishes': [dish],
                    'counts': [count],
                    'profit': profit,
                    'total_time_hours': [count * dish.time / 60],
                }

        # 尝试两个菜品的组合
        if self.max_slots >= 2:
            for dish1, dish2 in combinations(self.unlocked_dishes, 2):
                counts, profit = self._optimize_two_dishes(dish1, dish2)
                if profit > best_solution['profit']:
                    best_solution = {
                        'dishes': [dish1, dish2],
                        'counts': counts,
                        'profit': profit,
                        'total_time_hours': [
                            counts[0] * dish1.time / 60,
                            counts[1] * dish2.time / 60
                        ],
                    }

        # 计算食材需求和购买计划
        solutions: List[MenuSolution] = []
        required_ingredient_names: Set[str] = set()
        remaining_ingredients: Dict[str, int] = self.total_ingredients.copy()
        for idx in range(len(best_solution['dishes'])):
            dish: Dish = best_solution['dishes'][idx]
            count: int = best_solution['counts'][idx]
            ratio, remaining_ingredients = self._calc_bar_ratio(dish, count, remaining_ingredients)
            solutions.append(MenuSolution(dish, count, ratio))

            for ingredient_name in dish.ingredients.keys():
                required_ingredient_names.add(ingredient_name)

        return solutions, required_ingredient_names
        