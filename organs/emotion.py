import math
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from pkg.core import app
from plugins.Waifu.cells.config import ConfigManager

class EmotionEngine:
    """多维度情绪仿真引擎"""
    
    EMOTION_VECTOR_KEYS = [
        'happiness', 'sadness', 'anger', 
        'excitement', 'calmness', 'curiosity'
    ]
    
    def __init__(self, config_mgr: ConfigManager, launcher_id: str, ap: app.Application):
        self.ap = ap
        self.config = config_mgr.data
        self.base_response_delay = self.config.get('base_response_delay', 3)
        self.emotion_decay_window = self.config.get('emotion_decay_window', 2.0)
        self.logger = logging.getLogger("WaifuEmotion")
        
        # 基础情绪参数初始化
        self.base_vector = {k: 0.5 for k in self.EMOTION_VECTOR_KEYS}
        self.current_vector = self.base_vector.copy()
        self.last_update = datetime.now()
        self.decay_params = self._init_decay_parameters()
        self.emotion_log = []
        
        # 连接持久化系统
        self._storage_file = f"data/plugins/Waifu/data/emotion_{launcher_id}.json"
        self._load_persistent_state()

    def _init_decay_parameters(self) -> Dict[str, float]:
        """初始化情绪衰减参数"""
        return {
            'happiness': 0.98,
            'sadness': 0.95,
            'anger': 0.93,
            'excitement': 0.96,
            'calmness': 0.97,
            'curiosity': 0.99
        }

    def _load_persistent_state(self):
        """加载持久化情绪状态"""
        try:
            if not os.path.exists(self._storage_file):
                return
                
            with open(self._storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'current_vector' in data and 'timestamp' in data:
                    delta_time = (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds()
                    self._apply_decay(delta_time / 3600)
                    self.current_vector.update(data['current_vector'])
                    self.logger.info("情绪状态加载成功")
        except Exception as e:
            self.logger.error(f"情绪状态加载失败: {str(e)}")

    def save_persistent_state(self):
        """保存持久化情绪状态"""
        try:
            data = {
                'current_vector': self.current_vector,
                'timestamp': datetime.now().isoformat()
            }
            with open(self._storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"情绪状态保存失败: {str(e)}")

    def update_from_event(self, event_type: str, intensity: float, decay_mod: float = 1.0):
        """处理情绪事件"""
        current_time = datetime.now()
        delta_hours = (current_time - self.last_update).total_seconds() / 3600
        
        # 应用衰减后更新
        self._apply_decay(delta_hours)
        self._process_event(event_type, intensity, decay_mod)
        self.last_update = current_time
        
        # 记录情绪变更
        self.emotion_log.append({
            'timestamp': current_time.isoformat(),
            'event': event_type,
            'intensity': intensity,
            'vector': self.current_vector.copy()
        })
        self._trim_emotion_log()
        
        self.logger.debug(f"情绪更新：{event_type} 强度 {intensity:.2f}")

    def _apply_decay(self, delta_hours: float):
        """应用情绪衰减模型"""
        for emotion in self.EMOTION_VECTOR_KEYS:
            decay_rate = math.pow(self.decay_params[emotion], delta_hours)
            self.current_vector[emotion] = np.clip(
                self.current_vector[emotion] * decay_rate,
                0.0, 1.0
            )

    def _process_event(self, event_type: str, intensity: float, decay_mod: float):
        """处理不同类型情绪事件"""
        event_matrix = {
            'positive_message': {
                'happiness': +0.5 * intensity,
                'sadness': -0.3 * intensity,
                'calmness': +0.2
            },
            'negative_message': {
                'sadness': +0.6 * intensity,
                'anger': +0.4,
                'calmness': -0.5
            },
            'mentioned': {
                'excitement': +0.3,
                'curiosity': +0.4
            },
            'long_silence': {
                'calmness': +0.2,
                'excitement': -0.4
            },
            'emergency': {
                'excitement': +0.8,
                'calmness': -0.7,
                'anger': +0.5
            }
        }
        
        impacts = event_matrix.get(event_type, {})
        for emotion, delta in impacts.items():
            new_value = self.current_vector[emotion] + delta * decay_mod
            self.current_vector[emotion] = np.clip(new_value, 0.0, 1.0)
            
        # 确保情绪守恒律
        self._enforce_emotion_conservation()

    def _enforce_emotion_conservation(self):
        """情绪要素守恒约束"""
        MAX_SUM = 3.5
        current_sum = sum(self.current_vector.values())
        if current_sum > MAX_SUM:
            ratio = MAX_SUM / current_sum
            for k in self.current_vector:
                self.current_vector[k] *= ratio

    def _trim_emotion_log(self):
        """修剪情绪日志"""
        if len(self.emotion_log) > 500:
            self.emotion_log = self.emotion_log[-500:]

    def calculate_response_modifiers(self) -> Dict[str, float]:
        """计算响应调整参数"""
        excitement = self.current_vector['excitement']
        calmness = self.current_vector['calmness']
        happiness = self.current_vector['happiness']
        
        return {
            'probability_multiplier': self._response_probability_factor(excitement, calmness),
            'delay_modifier': self._response_delay_factor(excitement, calmness),
            'content_intensity': happiness * 0.7 + excitement * 0.3
        }

    def _response_probability_factor(self, excitement: float, calmness: float) -> float:
        """响应概率乘数"""
        base = np.tanh(excitement * 2) * 0.8 + 0.5
        calm_effect = np.power(calmness, 1.5) * 0.5
        return np.clip(base - calm_effect, 0.2, 1.8)

    def _response_delay_factor(self, excitement: float, calmness: float) -> float:
        """响应延迟修正"""
        excitement_delay = (1.5 - excitement) * self.config.base_response_delay
        calmness_effect = calmness * 0.8 * self.config.base_response_delay
        return excitement_delay + calmness_effect + np.random.normal(0, 0.1)

    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """获取当前情绪摘要"""
        primary_emotion = max(self.current_vector.items(), key=lambda x: x[1])[0]
        intensity_levels = {
            k: self._classify_intensity(v)
            for k, v in self.current_vector.items()
        }
        return {
            'primary_emotion': primary_emotion,
            'intensity_levels': intensity_levels,
            'current_vector': self.current_vector.copy(),
            'timestamp': datetime.now().isoformat()
        }

    def _classify_intensity(self, value: float) -> str:
        """情绪强度分级"""
        if value >= 0.8:
            return "extreme"
        elif value >= 0.6:
            return "high"
        elif value >= 0.4:
            return "moderate"
        elif value >= 0.2:
            return "low"
        return "neutral"

    def periodic_self_adjust(self):
        """周期性的自我调节"""
        delta = datetime.now() - self.last_update
        if delta.total_seconds() > 3600:  # 每小时自动调整
            self._apply_decay(delta.total_seconds() / 3600)
            self._natural_fluctuation()
            self.last_update = datetime.now()

    def _natural_fluctuation(self):
        """自然波动模型"""
        for emotion in self.current_vector:
            fluctuation = np.random.normal(0, 0.02)
            self.current_vector[emotion] = np.clip(
                self.current_vector[emotion] + fluctuation, 
                0.0, 1.0
            )
        self._enforce_emotion_conservation()

    def reset_emotion_state(self):
        """重置情绪状态到基线"""
        self.current_vector = self.base_vector.copy()
        self.emotion_log.clear()
        self.save_persistent_state()
        self.logger.info("情绪状态已重置")

    def calculate_emotional_impact(self) -> Dict[str, float]:
        """计算情绪对响应的影响"""
        excitement = self.current_vector['excitement']
        calmness = self.current_vector['calmness']
        
        return {
            'probability_boost': excitement * 0.5 - calmness * 0.3,
            'delay_reduction': (excitement - 0.5) * 0.7,
            'creativity_factor': self.current_vector['curiosity'] * 0.4
        }

    def get_response_modifiers(self) -> dict:
        excitement = self.current_vector['excitement']
        calmness = self.current_vector['calmness']
        return {
            'probability_multiplier': 0.5 + excitement - calmness * 0.3,
            'delay_factor': max(0.5, 1.5 - excitement * 0.8)
        }
