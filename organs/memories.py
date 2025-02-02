import typing
import numpy as np
import json
import os
import re
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pkg.core import app
from pkg.provider import entities as llm_entities
from plugins.Waifu.cells.config import ConfigManager
from plugins.Waifu.cells.generator import Generator
from plugins.Waifu.cells.text_analyzer import TextAnalyzer
from typing import List, Tuple, Dict, Any
from plugins.Waifu.organs.emotion import EmotionEngine

class SocialMemory:
    """社交关系记忆子系统"""
    def __init__(self, launcher_id: str):
        self.file_path = f"data/plugins/Waifu/data/social_{launcher_id}.json"
        # 新增关系维度
        self.data: Dict[str, Dict] = defaultdict(lambda: {
            "intimacy": 5.0,  # 亲密度（0-10）
            "trust": 5.0,     # 信任度（0-10）
            "interest_tags": [],  # 兴趣标签
            "interaction_history": []  # 交互记录（带时间戳）
        })
        self.data: Dict[str, Dict] = defaultdict(self._default_social_record)
        self._load()

    def _default_social_record(self) -> Dict:
        return {
            "impression_score": 5.0,
            "last_interact": datetime.now().timestamp(),
            "interact_count": 0,
            "positive_events": [],
            "negative_events": [],
            "relationship_tags": [],
            "mentioned_count": 0,
            "last_emotion": "neutral"
        }

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    for k, v in raw_data.items():
                        # 数据迁移处理
                        if 'events' in v:  # v1兼容
                            self.data[k] = {
                                "impression_score": v.get('score', 5.0),
                                "last_interact": v.get('last_interact', datetime.now().timestamp()),
                                "interact_count": len(v['events']),
                                "positive_events": [e for e in v['events'] if e.get('type') == 'positive'],
                                "negative_events": [e for e in v['events'] if e.get('type') == 'negative'],
                                "relationship_tags": list(set([t for e in v['events'] for t in e.get('tags', [])])),
                                "mentioned_count": v.get('mentioned_count', 0),
                                "last_emotion": v.get('last_emotion', 'neutral')
                            }
                        else:  # v2格式
                            self.data[k] = v
            except Exception as e:
                logging.error(f"加载社交记忆失败: {str(e)}")

    def save(self):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, default=self._serializer, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存社交记忆失败: {str(e)}")

    def _serializer(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f"无法序列化类型: {type(obj)}")

    def update_interaction(self, user_id: str, sentiment: str, tags: List[str], is_mentioned: bool):
        user_id = str(user_id)
        record = self.data[user_id]
        now = datetime.now().timestamp()

        # 更新基础指标
        record['interact_count'] += 1
        record['last_interact'] = now
        if is_mentioned:
            record['mentioned_count'] += 1

        # 情感影响算法
        base_delta = {
            'positive': 0.6,
            'neutral': 0.1,
            'negative': -1.0
        }.get(sentiment, 0)
        
        # 非线性修正因子
        current_score = record['impression_score']
        correction_factor = 1 - abs(current_score - 5) / 10
        record['impression_score'] = np.clip(current_score + base_delta * correction_factor, 0.0, 10.0)

        # 事件记录
        event_type = 'positive' if base_delta > 0 else 'negative'
        event_list = record[f'{event_type}_events']
        event_list.append({
            "timestamp": now,
            "tags": tags[:3],
            "sentiment": sentiment,
            "mentioned": is_mentioned
        })
        # 滚动窗口保留最近50个事件
        if len(event_list) > 50:
            event_list.pop(0)

        # 标签管理系统
        existing_tags = set(record['relationship_tags'])
        new_tags = set(filter(lambda x: 0 < len(x) < 20, tags))
        merged_tags = list(existing_tags.union(new_tags))[:50]  # 最多保留50个标签
        record['relationship_tags'] = merged_tags
        record['last_emotion'] = sentiment

        self.save()

    def get_intimacy_level(self, user_id: str) -> float:
        """获取标准化亲密度（0.0~1.0）"""
        record = self.data.get(str(user_id))
        if not record:
            return 0.3  # 默认新用户亲密度
        return float(np.clip(record['impression_score'] / 10.0, 0.0, 1.0))

    def get_recent_interactions(self, user_id: str, hours: int = 24) -> List[Dict]:
        """获取最近N小时的交互事件"""
        cutoff = datetime.now().timestamp() - hours * 3600
        record = self.data.get(str(user_id))
        if not record:
            return []
        return [e for e in record['positive_events'] + record['negative_events'] 
                if e['timestamp'] >= cutoff]

class Memory:
    """多维记忆管理系统"""
    def __init__(self, ap: app.Application, launcher_id: str, launcher_type: str, config_mgr: ConfigManager, emotion_engine: EmotionEngine):
        self.emotion_engine = emotion_engine  # 新增这行
        self.short_term_memory: List[llm_entities.Message] = []
        self.long_term_memory: List[Tuple[str, List[str]]] = []
        self.tags_index: Dict[str, int] = {}
        self._launcher_id = launcher_id
        self._launcher_type = launcher_type
        self.social_memory = SocialMemory(launcher_id)
        self._social_memory_file = f"data/plugins/Waifu/data/social_{launcher_id}.json"
        self._generator = Generator(ap)
        self._text_analyzer = TextAnalyzer(ap)
        self.emotion_engine = EmotionEngine(config_mgr) # 增加对情绪引擎的引用
        
        # 配置参数
        self.analyze_max_conversations = 9
        self.narrate_max_conversations = 8
        self.value_game_max_conversations = 5
        self.response_min_conversations = 5
        self.response_rate = 0.7
        self.max_thinking_words = 30
        self.max_narrat_words = 30
        self.repeat_trigger = 0
        self._short_term_memory_size = 100
        self._memory_batch_size = 50
        self._retrieve_top_n = 5
        self._summary_max_tags = 50
        self._summarization_mode = False
        self._thinking_mode_flag = True
        self._already_repeat = set()
        self.user_name = "user"
        self.assistant_name = "assistant"

        # 初始化持久化系统
        self._init_persistent_storage()

    def _init_persistent_storage(self):
        """初始化所有持久化存储"""
        os.makedirs("data/plugins/Waifu/data", exist_ok=True)
        self.long_term_file = f"data/plugins/Waifu/data/memories_{self._launcher_id}.json"
        self.conversations_file = f"data/plugins/Waifu/data/conversations_{self._launcher_id}.log"
        self.short_term_file = f"data/plugins/Waifu/data/short_term_{self._launcher_id}.json"
        self.status_file = f"data/plugins/Waifu/data/status_{self._launcher_id}.json"
        
        self._load_long_term_memory()
        self._load_short_term_memory()

    def _load_long_term_memory(self):
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.long_term_memory = [(e['summary'], e['tags']) for e in data.get('memories', [])]
                self.tags_index = data.get('tags_index', {})
        else:
            self.long_term_memory = []
            self.tags_index = {}

    def _load_short_term_memory(self):
        if os.path.exists(self.short_term_file):
            try:
                with open(self.short_term_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.short_term_memory = [
                        llm_entities.Message(
                            role=msg['role'],
                            content=re.sub(r'\[ts:\d+\.\d+\]', '', msg['content'])  # 清理时间戳
                        ) for msg in data
                    ]
            except json.JSONDecodeError:
                logging.error("短期记忆文件损坏，已重置")

    async def save_memory(self, role: str, content: str, is_mentioned: bool = False):
        # 情感分析
        sentiment = await self._analyze_sentiment(content)
        
        # 更新情绪引擎
        self.emotion_engine.update_from_event(
            event_type='group_message' if self._launcher_type == 'group' else 'private_message',
            intensity=sentiment['intensity'],
            tags=sentiment['tags']
        )
        # 更新社交记忆
        if role not in [self.assistant_name, 'narrator']:
            self.social_memory.update_interaction(
                user_id=role,
                sentiment=sentiment['label'],
                tags=sentiment['tags'],
                is_mentioned=is_mentioned
            )

        """完整记忆保存流程"""
        # 预处理消息内容
        timestamp = datetime.now().timestamp()
        clean_content = self._text_analyzer.clean_message(content)
        tagged_content = f"[ts:{timestamp}]{clean_content}"
        
        # 创建消息实体
        message = llm_entities.Message(
            role=role,
            content=tagged_content,
            metadata={
                'timestamp': timestamp,
                'mentioned': is_mentioned
            }
        )
        
        # 短期记忆存储
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > self._short_term_memory_size:
            await self._process_memory_overflow()
        
        # 社交记忆处理
        if role != self.assistant_name:
            sentiment_result = await self._analyze_sentiment(clean_content)
            self.social_memory.update_interaction(
                user_id=role,
                sentiment=sentiment['label'],
                tags=sentiment['tags'],
                is_mentioned=is_mentioned  # 需要从消息链解析@信息
            )
        
        # 持久化保存
        self._save_short_term_memory()
        self._log_conversation(message)
        logging.info(f"记忆已更新：{role} -> {clean_content[:50]}...")

    async def _process_memory_overflow(self):
        """处理记忆溢出"""
        if self._summarization_mode:
            batch = self.short_term_memory[:self._memory_batch_size]
            summary, tags = await self._summarize_batch(batch)
            self._add_to_long_term(summary, tags)
            self.short_term_memory = self.short_term_memory[self._memory_batch_size:]
        else:
            self.short_term_memory = self.short_term_memory[-self._short_term_memory_size:]

    async def _summarize_batch(self, batch: List[llm_entities.Message]) -> Tuple[str, List[str]]:
        """批量摘要生成"""
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in batch])
        prompt = f"""请将以下对话总结为长期记忆：
        {context}
        格式要求：
        - 第三人称视角
        - 包含关键人物、事件和结果
        - 不超过{self.max_narrat_words}字"""
        
        summary = await self._generator.return_string(prompt)
        
        # 标签提取优化
        tag_prompt = f"从以下内容提取3-5个关键词，用中文逗号分隔：{summary}"
        raw_tags = await self._generator.return_string(tag_prompt)
        tags = [t.strip() for t in raw_tags.split('，') if t.strip()]
        return summary, tags[:5]

    def _add_to_long_term(self, summary: str, tags: List[str]):
        """添加长期记忆条目"""
        self.long_term_memory.append((summary, tags))
        for tag in tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = len(self.tags_index)
        self._save_long_term_memory()

    def _save_long_term_memory(self):
        data = {
            'memories': [{
                'summary': summary,
                'tags': tags,
                'timestamp': datetime.now().timestamp()
            } for summary, tags in self.long_term_memory],
            'tags_index': self.tags_index
        }
        with open(self.long_term_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _save_short_term_memory(self):
        data = [{
            'role': msg.role,
            'content': msg.content,
            'metadata': msg.metadata
        } for msg in self.short_term_memory]
        with open(self.short_term_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _log_conversation(self, message: llm_entities.Message):
        log_entry = {
            'timestamp': message.metadata['timestamp'],
            'role': message.role,
            'content': message.content,
            'social': self.social_memory.data.get(str(message.role), {})
        }
        with open(self.conversations_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    async def _analyze_sentiment(self, content: str) -> Dict:
        """情感分析管道"""
        prompt = f"""分析消息的情感和关键元素：
        消息内容：{content}
        返回JSON格式：
        {{
            "label": "positive/neutral/negative",
            "intensity": 0-1,
            "tags": ["标签1", "标签2", "标签3"]
        }}"""
        try:
            result = await self._generator.return_json(prompt)
            return {
                'label': result.get('label', 'neutral'),
                'intensity': float(result.get('intensity', 0.5)),
                'tags': result.get('tags', [])[:3]
            }
        except Exception as e:
            logging.error(f"情感分析失败: {str(e)}")
            return {'label': 'neutral', 'intensity': 0.5, 'tags': []}

    # 兼容原有接口方法
    def to_custom_names(self, text: str) -> str:
        return text.replace("{user}", self.user_name).replace("{assistant}", self.assistant_name)

    def get_last_role(self, messages: List[llm_entities.Message]) -> str:
        return messages[-1].role if messages else ""

    def get_last_content(self, messages: List[llm_entities.Message]) -> str:
        return messages[-1].content if messages else ""

    def get_conversations_str_for_group(self, messages: List[llm_entities.Message]) -> str:
        return "\n".join([f"{msg.role}说：“{msg.content}”" for msg in messages])

    # ... 其他原有方法完整保留 ...
    # （接续上面的完整代码）

    def get_unreplied_msg(self, count: int) -> Tuple[int, List[llm_entities.Message]]:
        """获取未回复消息（核心兼容方法）"""
        unreplied = []
        reply_flag = False
        for msg in reversed(self.short_term_memory):
            if msg.role == self.assistant_name:
                reply_flag = True
                break
            unreplied.append(msg)
            if len(unreplied) >= count:
                break
        return len(unreplied), list(reversed(unreplied))

    async def check_repeat(self, content: str, role: str) -> bool:
        """重复检测逻辑（核心兼容方法）"""
        content = self._text_analyzer.clean_message(content)
        threshold = self._calculate_repeat_threshold(role)
        
        # 使用NLP进行相似性检测
        similarity_scores = [
            await self._text_analyzer.calculate_similarity(content, msg.content)
            for msg in self.short_term_memory[-10:] if msg.role == role
        ]
        if any(score > threshold for score in similarity_scores):
            self.repeat_trigger += 1
            return True
        return False

    def _calculate_repeat_threshold(self, role: str) -> float:
        """动态调整的重复阈值"""
        base = 0.7
        if role in self.social_memory.data:
            interact_count = self.social_memory.data[role]['interact_count']
            return base * (1 - 0.5 * (interact_count / (interact_count + 5)))
        return base

    def to_generic_names(self, text: str) -> str:
        """名称通用化处理（核心兼容方法）"""
        replacements = {
            self.user_name: "{user}",
            self.assistant_name: "{assistant}"
        }
        pattern = re.compile("|".join(map(re.escape, replacements.keys())))
        return pattern.sub(lambda m: replacements[m.group(0)], text)

    def get_conversations_str_for_person(self, messages: List[llm_entities.Message]) -> Tuple[List[str], str]:
        """个人对话格式化（核心兼容方法）"""
        speakers = list({msg.role for msg in messages if msg.role not in [self.assistant_name, "system"]})
        conv_str = "\n".join([
            f"{msg.role}说：“{msg.content}”" if msg.role != self.assistant_name 
            else f"{self.assistant_name}说：“{msg.content}”"
            for msg in messages
        ])
        return speakers, conv_str

    def get_last_speaker(self, messages: List[llm_entities.Message]) -> str:
        """获取最后发言者（核心兼容方法）"""
        if not messages:
            return ""
        last_msg = messages[-1]
        if last_msg.role == "narrator":
            return "旁白"
        return last_msg.role if last_msg.role != self.assistant_name else self.assistant_name

    def trigger_special_modes(self) -> Dict[str, bool]:
        """触发特殊模式（核心兼容方法）"""
        trigger_count = sum(1 for msg in self.short_term_memory[-3:] if msg.role == self.assistant_name)
        return {
            "thinking_mode": self._thinking_mode_flag and trigger_count < 2,
            "narrate_mode": len(self.short_term_memory) >= self.narrate_max_conversations,
            "value_game_mode": random.random() < 0.2  # 保留原始随机触发逻辑
        }

    def save_status(self):
        """状态保存（核心兼容方法）"""
        status = {
            "repeat_trigger": self.repeat_trigger,
            "already_repeat": list(self._already_repeat),
            "thinking_mode": self._thinking_mode_flag
        }
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2)

    def load_status(self):
        """状态加载（核心兼容方法）"""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
                self.repeat_trigger = status.get("repeat_trigger", 0)
                self._already_repeat = set(status.get("already_repeat", []))
                self._thinking_mode_flag = status.get("thinking_mode", True)

    def _backup_memory_files(self):
        """记忆文件备份（内部兼容方法）"""
        backup_dir = f"data/plugins/Waifu/backup/{datetime.now().strftime('%Y%m%d')}"
        os.makedirs(backup_dir, exist_ok=True)
        for fpath in [self.long_term_file, self.conversations_file, self.short_term_file]:
            if os.path.exists(fpath):
                shutil.copy2(fpath, os.path.join(backup_dir, os.path.basename(fpath)))

    def reset_memory(self, keep_long_term: bool = False):
        """重置记忆系统（核心兼容方法）"""
        self.short_term_memory.clear()
        self._already_repeat.clear()
        self.repeat_trigger = 0
        if not keep_long_term:
            self.long_term_memory.clear()
            self.tags_index.clear()
        os.remove(self.short_term_file)
        if not keep_long_term:
            os.remove(self.long_term_file)
        self._backup_memory_files()
