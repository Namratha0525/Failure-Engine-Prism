"""Alert fatigue suppression — prevents the same alert firing forever."""
import time, threading

class AlertState:
    def __init__(self, cooldown_seconds: int = 300, max_repeat: int = 3):
        self.cooldown = cooldown_seconds
        self.max_repeat = max_repeat
        self._last_time: float | None = None
        self._last_type: str | None = None
        self._repeat: int = 0
        self._lock = threading.Lock()

    def should_alert(self, failure_type: str, prob: float) -> tuple[bool, str]:
        """Return (should_fire, reason). HIGH confidence always fires."""
        is_high = prob >= 0.85
        with self._lock:
            now = time.time()
            if self._last_time is None:
                self._last_time, self._last_type, self._repeat = now, failure_type, 1
                return True, "FIRST_ALERT"

            elapsed = now - self._last_time
            same = failure_type == self._last_type

            if same and elapsed < self.cooldown and not is_high:
                self._repeat += 1
                if self._repeat > self.max_repeat:
                    return False, f"SUPPRESSED (same={failure_type}, {elapsed:.0f}s ago)"
                return True, f"REPEAT_{self._repeat}"

            self._last_time, self._last_type, self._repeat = now, failure_type, 1
            return True, "NEW_ALERT" if not same else "COOLDOWN_RESET"

    def reset(self):
        with self._lock:
            self._last_time = self._last_type = None
            self._repeat = 0

# Singleton
_alert_state: AlertState | None = None

def get_alert_state() -> AlertState:
    global _alert_state
    if _alert_state is None:
        _alert_state = AlertState()
    return _alert_state

def init_alert_state(cooldown: int, max_repeat: int):
    global _alert_state
    _alert_state = AlertState(cooldown, max_repeat)
    return _alert_state
