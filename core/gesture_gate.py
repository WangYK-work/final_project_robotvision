import math


class PinchDrawGate:
    """Right-hand pinch starts a stroke; releasing it ends the stroke."""

    def __init__(self, start_ratio=0.45, stop_ratio=0.65, stable_frames=2):
        self.start_ratio = start_ratio
        self.stop_ratio = stop_ratio
        self.stable_frames = stable_frames
        self.active = False
        self.near_frames = 0

    def update(self, hand):
        if hand is None:
            self.active = False
            self.near_frames = 0
            return False

        landmarks = hand["landmarks"]
        palm_width = max(1.0, math.dist(landmarks[5], landmarks[17]))
        ratio = math.dist(landmarks[4], landmarks[8]) / palm_width
        if self.active:
            if ratio >= self.stop_ratio:
                self.active = False
                self.near_frames = 0
        else:
            self.near_frames = self.near_frames + 1 if ratio <= self.start_ratio else 0
            if self.near_frames >= self.stable_frames:
                self.active = True
                self.near_frames = 0
        return self.active


def _angle(a, b, c):
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    length = math.hypot(*u) * math.hypot(*v)
    if length < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / length))))


def pinch_ratio(hand):
    p = hand["landmarks"]
    return math.dist(p[4], p[8]) / max(1.0, math.dist(p[5], p[17]))


def classify_number(hand):
    """Palm-facing Chinese numbers; geometry is independent of image rotation."""
    if hand is None:
        return None
    p = hand["landmarks"]
    palm = math.dist(p[5], p[17])
    if palm < 8:
        return None
    extended = [
        _angle(p[mcp], p[mcp + 1], p[mcp + 3]) >= 155
        and math.dist(p[mcp + 3], p[0]) > 1.12 * math.dist(p[mcp + 1], p[0])
        for mcp in (5, 9, 13, 17)
    ]
    # Seven is a gathering of three fingertips away from the palm, not a fist.
    gathered = max(math.dist(p[a], p[b]) for a, b in ((4, 8), (4, 12), (8, 12))) < .38 * palm
    forward = all(math.dist(p[tip], p[0]) > 1.10 * math.dist(p[mcp], p[0])
                  for tip, mcp in ((8, 5), (12, 9)))
    if gathered and forward and not any(extended[2:]):
        return 7
    thumb = (_angle(p[1], p[2], p[3]) >= 145 and _angle(p[2], p[3], p[4]) >= 145
             and math.dist(p[4], p[5]) > .45 * palm and math.dist(p[4], p[17]) > 1.05 * palm)
    if thumb and extended == [False, False, False, True]:
        return 6
    if thumb and all(extended):
        return 5
    if not thumb:
        patterns = { (True, False, False, False): 1, (True, True, False, False): 2,
                     (True, True, True, False): 3, (True, True, True, True): 4 }
        return patterns.get(tuple(extended))
    return None


class NumberGestureSelector:
    """Time-based confirmation with a neutral release before repeating a number."""
    def __init__(self):
        self.candidate = None
        self.since = None
        self.latched = None
        self.neutral_since = None
        self.progress = 0.0

    def update(self, hand, now):
        return self.update_number(classify_number(hand), now)

    def update_number(self, number, now):
        if number is None:
            self.candidate, self.since, self.progress = None, None, 0.0
            if self.neutral_since is None:
                self.neutral_since = now
            if now - self.neutral_since >= .25:
                self.latched = None
            return None
        self.neutral_since = None
        if number != self.candidate:
            self.candidate, self.since = number, now
        duration = 1.2 if number == 3 else .5
        self.progress = min(1.0, (now - self.since) / duration)
        if self.progress >= 1 and number != self.latched:
            self.latched = number
            return number
        return None


class ViewGestureGate:
    """Latch one pinch until both fingertips release; reject ambiguous pinches."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = self.candidate = None
        self.frames = 0
        self.wait_release = True

    def update(self, hand):
        if hand is None:
            self.reset()
            self.wait_release = False
            return None
        p = hand['landmarks']
        palm = max(1.0, math.dist(p[5], p[17]))
        index, middle = (math.dist(p[4], p[tip]) / palm for tip in (8, 12))
        released = index >= .65 and middle >= .65
        if self.wait_release:
            if released:
                self.wait_release = False
            return None
        if self.mode:
            ratio = index if self.mode == 'PAN' else middle
            other = middle if self.mode == 'PAN' else index
            if ratio >= .65 or other <= .45:
                self.reset()
                self.wait_release = not released
                return None
            return self.mode
        candidate = 'PAN' if index <= .45 and middle >= .65 else 'ZOOM' if middle <= .45 and index >= .65 else None
        self.frames = self.frames + 1 if candidate and candidate == self.candidate else 1 if candidate else 0
        self.candidate = candidate
        if self.frames >= 2:
            self.mode = candidate
        return self.mode
