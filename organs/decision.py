import math
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import numpy as np
from pkg.core import app
from plugins.Waifu.cells.config import ConfigManager
from plugins.Waifu.organs.memories import Memory
from plugins.Waifu.cells.generator import Generator
from plugins.Waifu.organs.emotion import EmotionEngine

class DecisionEngine:
    
    def __init__(self, ap: app.Application, config: dict, generator: Generator, emotion_engine: EmotionEngine):
        self.emotion_engine = emotion_engine  # 确保存储emotion实例
        self.ap = ap
        self.config = config
        self.generator = generator
        self.model_call_count = 0
        self.last_model_process = 0.0
        self.emotion_cooldown = 0.0
        self.active_conversations: Dict[str, float] = {}  # {user_id: expiry_time}
        self.logger = logging.getLogger("WaifuDecision")
        self.base_threshold = config.get('decision_weights', {}).get('base_threshold', 0.65)
        self.b_class_threshold = config.get('decision_weights', {}).get('b_class_threshold', 0.4)
        self.last_trigger_user: Optional[str] = None
        self.last_trigger_time: float = 0.0

    async def compute_reply_probability(
        self, 
        message_chain: list,
        current_time: float,
        memory: Memory,
        group_member_count: int, emotion_impact: Dict[str, float]) -> Tuple[bool, float, float]:
        """消息处理全流程决策"""
        try:
            # 核心决策流水线
            base_weights = await self._compute_base_weights(message_chain, current_time, memory)
            dynamic_weights = await self._compute_dynamic_weights(message_chain, memory, current_time)
            emotion_factor = self._compute_emotion_impact(current_time)
            
            # 组合权重
            total_score = self._combine_weights(base_weights, dynamic_weights, emotion_factor)
            final_delay = self._calculate_response_delay(base_weights, emotion_factor)
            
            # 多重触发机制
            should_reply = self._check_triggers(
                base_weights=base_weights,
                dynamic_weights=dynamic_weights,
                total_score=total_score,
                message_chain=message_chain
            )
            
            # 群体智能抑制
            if group_member_count > 1:
                should_reply = self._apply_group_suppression(should_reply, total_score, group_member_count)

            if should_reply:
                sender = self._extract_sender(message_chain)
                self.last_trigger_user = sender
                self.last_trigger_time = current_time
            
            return should_reply, total_score, final_delay
            
        except Exception as e:
            self.logger.error(f"决策流程异常: {str(e)}", exc_info=True)
            return False, 0.0, 0.0

    def _compute_base_weights(self, message_chain, current_time, memory):
        emotion_mod = self.emotion_engine.get_response_modifiers()
        """核心基础权重计算"""
        weights = {
            'keyword': self._keyword_weight(message_chain, memory),
            'mention': self._mention_weight(message_chain),
            'silence': self._silence_compensation(current_time, memory),
            'time': self._time_of_day_weight(current_time),
            'social': self._social_affinity_weight(message_chain, memory),
            'emotion_base': self._base_emotion_factor(memory),
            'heat': self._group_heat_factor(message_chain, memory, current_time),
            'emotion': self.emotion_engine.get_emotional_impact()['probability_boost'],
            'emotion_prob': emotion_mod['probability_multiplier'],
            'emotion_delay': emotion_mod['delay_factor']
        }
        return weights

    async def _compute_dynamic_weights(
        self,
        message_chain: list,
        memory: Memory,
        current_time: float
    ) -> Dict[str, float]:
        """动态权重计算（含大模型调用）"""
        # 大模型调用频率控制
        if current_time - self.last_model_process > self.cache.model_process_interval:
            semantic_weight = await self._semantic_relevance_weight(message_chain, memory)
            continuity_weight = await self._conversation_continuity_weight(memory)
            self.last_model_process = current_time
        else:
            semantic_weight = self.cache.last_semantic_cache
            continuity_weight = self.cache.continuity_weight_cache
            
        return {
            'semantic': semantic_weight,
            'continuity': continuity_weight,
            'long_term_memory': self._long_term_memory_weight(memory),
            'cooldown': self._cooldown_factor(current_time)
        }

    def _keyword_weight(self, message_chain: list, memory: Memory) -> float:
        """A类权重：关键词检测"""
        content = ' '.join([str(c) for c in message_chain])
        matched = sum(1 for kw in memory.social_memory.keyword_response_keywords if kw in content)
        return min(matched * self.cache.keyword_weight, 1.0)

    def _mention_weight(self, message_chain: list) -> float:
        """A类权重：@提到检测"""
        is_mentioned = any(c.type == 'mention' for c in message_chain)
        return self.cache.mention_weight * float(is_mentioned)

    def _silence_compensation(self, current_time: float, memory: Memory) -> float:
        """改进后的拟人化沉默补偿"""
        if memory.last_response_time == 0:
            return 1.0
        
        silence_duration = current_time - memory.last_response_time
        compensation = min(silence_duration / 600, 1.0)  # 10分钟达到最大补偿
        
        # 引入随机波动模拟人类耐心
        jitter = np.random.normal(0, 0.1)
        return np.clip(compensation + jitter, 0.0, 1.5)  # 允许最高150%补偿

    def _time_of_day_weight(self, current_time: float) -> float:
        """A类权重：时间因子"""
        dt = datetime.fromtimestamp(current_time)
        # 使用正弦波模拟生物钟
        hour = dt.hour + dt.minute / 60.0
        base = 0.5 * math.sin((hour - 8) * math.pi / 12) + 0.5

        # 深夜时段特殊处理
        if 0 <= hour < 2:
            return 0.3
        elif 2 <= hour < 8:
            return 0.0
        elif 12 <= hour < 13:
            return 0.5

        # 正常时间段计算
        return 0.5 * math.sin((hour - 8) * math.pi / 12) + 0.5

    def _social_affinity_weight(self, message_chain: list, memory: Memory) -> float:
        sender = self._extract_sender(message_chain)
        if not sender or sender == memory.assistant_name:
            return 0.0

        intimacy = memory.social_memory.get_intimacy_level(sender)
        
        # 连续对话加成（新增部分）
        if (sender == self.last_trigger_user and 
            (time.time() - self.last_trigger_time) < 600):
            intimacy = min(intimacy * 1.5, 1.0)
        
        return math.pow(intimacy, 2.5) * self.decision_weights.get('social_weight', 0.7)

    def _group_heat_factor(self, message_chain: list, memory: Memory, current_time: float) -> float:
        """群体热度抑制因子"""
        # 时间窗口：5分钟内
        time_window = current_time - 300
        recent_messages = [
            msg for msg in memory.short_term_memory
            if hasattr(msg, 'metadata') and msg.metadata.get('timestamp', 0) > time_window
        ]
        
        participant_count = len({msg.role for msg in recent_messages if msg.role != memory.assistant_name})
        message_rate = len(recent_messages) / 300  # 每秒消息率
        
        heat_factor = 1.0 / (1 + math.exp(0.5 * (message_rate - 3)))
        return max(0.2, heat_factor)

    async def _semantic_relevance_weight(self, message_chain: list, memory: Memory) -> float:
        """B类权重：语义相关性（大模型调用）"""
        context = self._get_recent_context(memory)
        query = ' '.join(str(c) for c in message_chain)
        
        prompt = f"""综合评估下列对话与用户兴趣的相关性：
        用户兴趣列表：{memory.cards.interest_keywords}
        近期对话：{context[-500:]}
        当前消息：{query}
        
        返回JSON格式：
        {{
            "relevance": 0-1的评分,
            "reason": "简要理由"
        }}"""
        
        try:
            response = await self._generator.return_json(prompt)
            score = max(0.0, min(1.0, float(response.get('relevance', 0.0))))
            self.cache.last_semantic_cache = score
            return score * self.cache.semantic_weight
        except Exception as e:
            self.logger.warning(f"语义分析失败: {str(e)}")
            return 0.0

    def _long_term_memory_weight(self, memory: Memory) -> float:
        """B类权重：长期记忆唤醒"""
        recent_tags = Counter()
        for msg in memory.short_term_memory[-5:]:
            if hasattr(msg, 'metadata') and 'semantic_tags' in msg.metadata:
                recent_tags.update(msg.metadata['semantic_tags'])
                
        overlap = sum(
            1 for tag in recent_tags 
            if tag in memory.long_term_memory.tags_index
        )
        return min(overlap / 3.0, 1.0) * self.cache.long_term_weight

    async def _conversation_continuity_weight(self, memory: Memory) -> float:
        """B类权重：对话连续性（大模型调用）"""
        if not memory.short_term_memory:
            return 0.0
            
        last_response = memory.short_term_memory[-1].content
        context = self._get_recent_context(memory)
        
        prompt = f"""判断当前对话是否需要继续：
        最近对话：{context[-300:]}
        最后回复：“{last_response}”
        
        返回JSON格式：
        {{
            "should_continue": 布尔值,
            "confidence": 0-1的置信度
        }}"""
        
        try:
            response = await self._generator.return_json(prompt)
            if response.get('should_continue', False):
                confidence = max(0.0, min(1.0, float(response.get('confidence', 0.7))))
                self.cache.continuity_weight_cache = confidence * self.cache.continuity_weight
            else:
                self.cache.continuity_weight_cache = 0.0
            return self.cache.continuity_weight_cache
        except Exception as e:
            self.logger.warning(f"连续性分析失败: {str(e)}")
            return 0.0

    def _combine_weights(self, base_weights, dynamic_weights, emotion_impact):
        """从情绪引擎获取当前情绪"""
        emotion_vector = self.emotion_engine.get_emotional_state_summary()
        
        emotion_prob_boost = emotion_impact.get('probability_boost', 0.0)
        emotion_delay_factor = emotion_impact.get('delay_factor', 1.0)

        ['current_vector']
        return (emotion_vector['excitement'] - emotion_vector['calmness']) * 2.0

    def _combine_weights(self, base_weights, dynamic_weights, emotion_impact):
        # 获取情绪影响因子
        emotion_prob_boost = emotion_impact.get('probability_boost', 0.0)
        emotion_delay_factor = emotion_impact.get('delay_factor', 1.0)
        
        # 组合基础权重
        base_total = (
            base_weights['keyword'] * 0.3 +
            base_weights['mention'] * 0.2 +
            base_weights['silence'] * 0.15 +
            base_weights['social'] * 0.35
        )
        
        # 组合动态权重
        dynamic_total = (
            dynamic_weights['semantic'] * 0.5 +
            dynamic_weights['continuity'] * 0.3 +
            dynamic_weights['long_term_memory'] * 0.2
        )
        
        # 情绪影响最终得分
        final_score = (base_total * 0.6 + dynamic_total * 0.4) * (1 + emotion_prob_boost)
        return max(0.0, min(final_score, 2.0))

    def _calculate_response_delay(self, base_weights, emotion_impact):
        # 基础延迟计算
        base_delay = self.config.get('base_response_delay', 3)
        
        # 情绪影响延迟
        emotion_delay_factor = emotion_impact.get('delay_factor', 1.0)
        adjusted_delay = base_delay * emotion_delay_factor
        
        # 添加随机波动
        jitter = np.random.normal(0, 0.3)
        return max(0.5, adjusted_delay + jitter)

    def _check_triggers(
        self,
        base_weights: Dict[str, float],
        dynamic_weights: Dict[str, float],
        total_score: float,
        message_chain: list
    ) -> bool:
        """复杂触发条件判断"""
        # A类基础触发
        if (total_score >= self.cache.base_threshold or 
            (base_weights['keyword'] >= self.cache.keyword_threshold and 
             base_weights['mention'] >= self.cache.mention_threshold)):
            return True
            
        # B类独立触发
        b_triggers = [
            dynamic_weights['semantic'] >= self.cache.semantic_threshold,
            dynamic_weights['continuity'] >= self.cache.continuity_threshold,
            dynamic_weights['long_term_memory'] >= self.cache.long_term_threshold
        ]
        if any(b_triggers) and total_score >= self.cache.b_class_threshold:
            return True
            
        # 紧急响应触发
        if self._check_emergency_trigger(message_chain):
            return True
            
        return False

    def _apply_group_suppression(self, current_decision, total_score, member_count):
        """群体场景压制逻辑"""
        if member_count < 3:
            return current_decision
            
        suppression_factor = 1.0 - (0.15 * (member_count - 2))
        adjusted_score = total_score * suppression_factor
        return adjusted_score >= self.cache.group_threshold

    def _check_emergency_trigger(self, message_chain: list) -> bool:
        """紧急关键词触发（安全机制）"""
        emergency_keywords = {
            '紧急求助', '救命', 'urgent', 'help',
            '危险', '着火', 'emergency', 'accident'
        }
        content = ' '.join(str(c) for c in message_chain).lower()
        return any(kw in content for kw in emergency_keywords)

    # ----------------------
    # 工具方法
    # ----------------------
    def _extract_sender(self, message_chain: list) -> Optional[str]:
        """从消息链中提取发送者"""
        for element in message_chain:
            if hasattr(element, 'sender'):
                return str(element.sender)
        return None
        
    def _get_recent_context(self, memory: Memory, lookback: int = 8) -> str:
        """获取近期对话上下文"""
        return "\n".join([
            f"{msg.role}: {msg.content}" 
            for msg in memory.short_term_memory[-lookback:]
        ])

    def _base_emotion_factor(self, memory: Memory) -> float:
        """基础情绪指数计算"""
        emotion_vector = memory.emotion_state.current_vector
        excitement = emotion_vector.get('excitement', 0.5)
        calm = emotion_vector.get('calmness', 0.5)
        return (excitement - calm) * 2.0

    def _cooldown_factor(self, current_time: float) -> float:
        """冷却时间因子"""
        time_since_last = current_time - self.cache.last_response_time
        if time_since_last < self.cache.response_cooldown:
            return max(0.2, 1 - (time_since_last / self.cache.response_cooldown))
        return 1.0
