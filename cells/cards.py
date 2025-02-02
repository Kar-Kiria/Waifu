import typing
import re
from pkg.core import app
from plugins.Waifu.cells.config import ConfigManager
import numpy as np


class SocialRelationship:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.intimacy = 5.0  # 初始亲密度
        self.trust_level = 5.0  # 初始信任度
        self.shared_topics = []  # 共同话题列表
        self.interaction_count = 0  # 互动次数
        
    def to_dict(self):
        return {
            "intimacy": self.intimacy,
            "trust": self.trust_level,
            "shared_topics": self.shared_topics,
            "interaction_count": self.interaction_count
        }

class Cards:

    ap: app.Application

    def __init__(self, ap: app.Application):
        self.ap = ap
        self._user_name = "user"
        self._assistant_name = "assistant"
        self._language = ""
        self._profile = []
        self._skills = []
        self._background = []
        self._output_format = []
        self._rules = []
        self._manner = ""
        self._memories = []
        self._prologue = ""
        self._additional_keys = {}
        self._has_preset = True
        self._interest_keywords = []  # 新增兴趣标签字段
        self._social_network = {}  # 新增社交网络字典，key为用户ID
        self._personality_vector = np.zeros(5)  # 新增五维个性向量 (开放度,责任度,外向度,亲和度,神经质)

 

    async def load_config(self, character: str, launcher_type: str):
        if character == "off":
            self._has_preset = False
            return
        self._has_preset = True

        config = ConfigManager(f"data/plugins/Waifu/cards/{character}", f"plugins/Waifu/templates/default_{launcher_type}")
        await config.load_config(completion=False)
        self._user_name = config.data.get("user_name", "用户")
        self._assistant_name = config.data.get("assistant_name", "助手")
        self._language = config.data.get("language", "简体中文")
        self._profile = config.data.get("Profile", [])
        if isinstance(self._profile, list) and self._assistant_name != "助手":
            self._profile = [f"你是{self._assistant_name}。"] + self._profile
        self._skills = config.data.get("Skills", [])
        self._background = config.data.get("Background", [])
        if isinstance(self._background, list) and launcher_type == "person" and self._assistant_name != "助手" and self._user_name != "用户":
            self._background = self._background + [f"你是{self._assistant_name}，用户是{self._user_name}。"]
        self._rules = config.data.get("Rules", [])
        self._prologue = config.data.get("Prologue", "")

        # 新增兴趣标签加载
        self._interest_keywords = config.data.get("Interests", [])
        
        # 加载个性五维
        personality_config = config.data.get("Personality", {})
        self._init_personality_vector(personality_config)

        # 从记忆系统加载社交关系数据
        await self._load_social_network(launcher_type)

        # Collect additional keys
        predefined_keys = {"user_name", "assistant_name", "language", "Profile", "Skills", "Background", "Rules", "Prologue", "max_manner_change", "value_descriptions"}
        self._additional_keys = {key: value for key, value in config.data.items() if key not in predefined_keys}

    def set_memory(self, memories: typing.List[str]):
        self._memories = memories

    def set_manner(self, manner: str):
        self._manner = manner

    def get_background(self) -> str:
        return self._format_value(self._background)

    def get_profile(self) -> str:
        return self._format_value(self._profile)

    def get_manner(self) -> str:
        return self._manner

    def get_prologue(self) -> str:
        return self._format_value(self._prologue)

    def get_rules(self) -> str:
        init_parts = []
        if self._rules:
            init_parts.append(self._format_value(self._rules, "你必须遵守"))
        if self._manner:
            init_parts.append(self._format_value(self._manner, "你必须遵守"))
        if self._language:
            init_parts.append(f"你必须用默认的{self._language}与我交谈。")
        return "".join(init_parts)

    def generate_system_prompt(self) -> str:
        return self._format_value(self._collect_prompt_sections())

    def _collect_prompt_sections(self) -> typing.List[typing.Tuple[str, typing.Any]]:
        sections = []

        # 逐一检查每个部分，如果非空，则添加到 sections 中
        if self._profile:
            sections.append(("Profile", self._profile))
        if self._skills:
            sections.append(("Skills", self._skills))
        if self._background:
            sections.append(("Background", self._background))
        if self._memories:
            sections.append(("Memories", self._memories))
        rules = self.get_rules()
        if rules:
            sections.append(("Rules", rules))

        # 添加额外的 key，如果 value 非空
        for key, value in self._additional_keys.items():
            if value:  # 检查值是否为空
                sections.append((key, value))

        return sections

    def _ensure_punctuation(self, text: str | None) -> str:
        if isinstance(text, str):
            # 定义中英文标点符号
            punctuation = r"[。.，,？?；;]"
            # 如果末尾没有标点符号，则添加一个句号
            if not re.search(punctuation + r"$", text):
                return text + "。"
            return text
        else:
            return ""

    def _format_value(self, value: typing.Any, prefix: str = "", link: str = "") -> str:
        """
        统一处理 list、dict、str 等类型，并支持嵌套结构。
        """
        if isinstance(value, dict):
            # 处理字典，递归格式化
            formatted = []
            for k, v in value.items():
                formatted.append(f"{prefix}{k}:")
                formatted.append(self._format_value(v, prefix, link))
            return link.join(formatted)

        elif isinstance(value, list):
            # 处理列表，递归格式化每个元素
            formatted = []
            for item in value:
                formatted.append(self._format_value(item, prefix, link))
            return link.join(formatted)

        elif isinstance(value, str):
            # 处理字符串，确保末尾标点
            return prefix + self._ensure_punctuation(value)

        else:
            # 其他类型，强制转换为字符串
            return prefix + str(value)
        
    def _init_personality_vector(self, personality_data: dict):
        default = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }
        traits = {**default, **personality_data}
        self._personality_vector = np.array([
            traits["openness"],
            traits["conscientiousness"],
            traits["extraversion"],
            traits["agreeableness"],
            traits["neuroticism"]
        ], dtype=np.float32)  # 明确数据类型

    # 新增社交网络加载方法
    async def _load_social_network(self, launcher_type: str):
        social_file = f"data/plugins/Waifu/data/social_{launcher_type}.json"
        try:
            with open(social_file, 'r') as f:
                data = json.load(f)
                for user_id, rel_data in data.items():
                    rel = SocialRelationship(user_id)
                    rel.intimacy = rel_data.get('intimacy', 5.0)
                    rel.trust_level = rel_data.get('trust', 5.0)
                    rel.shared_topics = rel_data.get('shared_topics', [])
                    rel.interaction_count = rel_data.get('interaction_count', 0)
                    self._social_network[user_id] = rel
        except FileNotFoundError:
            pass

    # 新增获取兴趣标签接口
    @property
    def interest_keywords(self) -> list:
        return self._interest_keywords

    # 新增社交关系访问方法
    def get_relationship(self, user_id: str) -> SocialRelationship:
        return self._social_network.get(user_id, SocialRelationship(user_id))

    # 新增个性相似度计算方法（供决策引擎调用）
    def calculate_personality_similarity(self, other_vector: np.ndarray) -> float:
        return float(np.dot(self._personality_vector, other_vector))
