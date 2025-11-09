import time
import json
from typing import Dict, List, Set
import re
import os
import numpy as np
from maa.custom_action import CustomAction
from maa.context import Context


class IdolTrainingMainProcess(CustomAction):
    """
    custom_action_param中传入一个字典：{
    "config_path": config_path,
    "training_plan": {
        "names": "name1,name2"，
        "乐感": “0,0“,
        "体能": “1,0“,
        "表现": “2,2“,
        "情绪": “0,0“,
        "协调": “0,0“
    }}
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        training_param: Dict = json.loads(argv.custom_action_param)
        config_path: str = os.path.join(os.getcwd(), training_param["config_path"])
        portraits = [[100, 142, 1, 1], [100, 242, 1, 1], [100, 342, 1, 1],
                     [100, 442, 1, 1], [100, 542, 1, 1], [100, 642, 1, 1]]
        project = ['乐感', '体能', '协调', '情绪', '表现']
        analysed_plan = self.plan_analyse(training_param["training_plan"])
        # 定义基本任务
        context.override_pipeline({
            "character_list_slide": {
                "action": {
                    "type": "Swipe",
                    "param": {
                        "begin": [98, 641, 0, 0],
                        "end": [98, 34, 0, 0],
                        "duration": 2000,
                        "end_hold": 1000
                    }
                }
            },
            "character_name_reco": {
                "recognition": {
                    "type": "Custom",
                    "param": {
                        "custom_recognition": "RobustlyNameRecognize",
                        "custom_recognition_param": config_path,
                        "roi": [855, 88, 108, 43]
                    }
                }
            },
            "skip_performance": {
                "inverse": True,
                "recognition": {
                    "type": "TemplateMatch",
                    "param": {
                        "roi": [617, 632, 54, 54],
                        "template": "偶像训练/跳过演出.png"
                    }
                },
                "action": "Click"
            },
            "remain_num_reco": {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": [875, 550, 52, 22],
                        "expected": "\\d+/\\d+"
                    }
                }
            },
            "double_click_to_train": {
                "pre_delay": 0,
                "action": {
                    "type": "Click",
                    "param": {
                        "target": "execute_plan"
                    }
                },
                "post_delay": 0
            },
            "character_did_not_obtain": {
                "recognition": {
                    "type": "OCR",
                    "param": {
                        "roi": [916, 606, 181, 60],
                        "expected": ["未获得", "者"]
                    }
                }
            }
        })
        # 识别剩余次数
        remain_detail = context.run_recognition("remain_num_reco",
                                                context.tasker.controller.post_screencap().wait().get())
        if remain_detail:
            remains = int((remain_detail.best_result.text if remain_detail.best_result is not None
                           else remain_detail.filterd_results[0].text).split("/")[0])
        else:
            remains = 0

        '''训练进程'''
        context.run_task("skip_performance")
        for name, plan in self.trim_exceeded_plan(analysed_plan, remains).items():
            recorded_characters: Set[str] = set()
            training_done = False
            is_last_page = False
            while not is_last_page:
                for portrait in portraits:
                    context.run_task("click_portrait", {
                        "action": {
                            "type": "Click",
                            "param": {
                                "target": portrait
                            }
                        }
                    })
                    name_task_detail = context.run_recognition("character_name_reco",
                                                               context.tasker.controller.post_screencap().wait().get()
                                                               )
                    if name_task_detail is None:
                        return CustomAction.RunResult(success=False)
                    else:
                        name_detail = name_task_detail.best_result

                    if name_detail is None:  # 角色名未识别到，继续识别下一个
                        continue
                    if name_detail.text in recorded_characters:  # 角色名已记录，表明本页是最后一页
                        is_last_page = True
                        continue
                    if name_detail.text != name:  # 角色名不正确，记录后继续识别
                        recorded_characters.add(name_detail.text)
                        continue

                    # 角色正确，开始训练
                    current_character_detail = context.run_task("character_did_not_obtain")
                    if current_character_detail is None or current_character_detail.status.failed:
                        # 角色未解锁，跳过
                        training_done = True
                        break
                    # 由于训练完后会退出重进，因此不需要记录目标角色
                    for idx in range(len(plan)):
                        if plan[idx] == 0:
                            continue
                        # 双击训练
                        context.run_task("execute_plan", {
                            "recognition": {
                                "type": "TemplateMatch",
                                "param": {
                                    "roi": [771, 587, 463, 81],
                                    "template": f"偶像训练/{project[idx]}.png"
                                }
                            }
                        })
                        context.run_task("double_click_to_train")
                        context.run_task("double_click_to_train")
                    training_done = True
                    break
                if training_done:  # 若该角色的训练已结束，重进训练界面
                    context.run_task("返回上级菜单")
                    context.run_task("训练室_修正者")
                    time.sleep(3)
                    break
                else:
                    context.run_task("character_list_slide")  # 否则翻页

        '''训练进程结束'''
        context.run_task("直接返回主菜单")
        return CustomAction.RunResult(success=True)

    @staticmethod
    def plan_analyse(plan: Dict[str, str]) -> Dict[str, List[int]]:
        analysed: Dict[str, List[int]] = {}
        names: List[str] = re.findall(r"[\u4e00-\u9fa5]+", plan.pop("names"))  # 通过汉字识别角色名
        projects = []
        for project, nums in plan.items():  # 导出单一角色的训练项目
            nums = re.findall(r"\d+", nums)
            projects.append([int(num) for num in nums])
        single_plan = list(zip(*projects))
        for idx in range(len(single_plan)):
            analysed[names[idx]] = single_plan[idx]
        return analysed

    @staticmethod
    def trim_exceeded_plan(analysed_plan: Dict[str, List[int]], remain: int) -> Dict[str, List[int]]:
        """去除超出上限的计划"""
        trimmed: Dict[str, List[int]] = {}
        done = False
        for name, plan in analysed_plan.items():
            plan_num: int = np.array(plan, dtype=np.int64).sum().item()  # 计算当前角色的训练总次数
            if plan_num < remain:  # 将全部计划加入结果，更新剩余次数后继续计算下一个角色
                trimmed[name] = plan
                remain -= plan_num
            elif plan_num == remain:  # 将全部计划加入结果，停止统计
                trimmed[name] = plan
                break
            else:
                trimmed[name] = []
                for train_num in plan:  # 按照 '乐感', '体能', '协调', '情绪', '表现' 的顺序筛选可以加入的计划
                    if train_num < remain:  # 将当前项目加入计划，更新剩余次数后继续计算下一个项目
                        trimmed[name].append(train_num)
                        remain -= train_num
                    else:
                        trimmed[name].append(remain)  # 加入全部剩余次数
                        trimmed[name].extend(np.zeros(5 - len(trimmed[name]), dtype=np.int64).tolist())  # 填补空位
                        done = True
                        break
                if done:
                    break
        return trimmed
