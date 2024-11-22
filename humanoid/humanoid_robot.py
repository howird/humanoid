import os

import sapien
import numpy as np

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

MJCF_PATH = f"{os.path.join(os.path.dirname(__file__), 'assets/mocap_humanoid.xml')}"

@register_agent()
class HumanoidRobot(BaseAgent):
    uid = "humanoid_robot"
    mjcf_path = MJCF_PATH
    group_collisions_by_depth = True
