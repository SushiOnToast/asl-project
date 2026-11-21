class FingerSpeller:
    def __init__(self, confidence_threshold=0.8, hold_frames=20):
        self.confidence_threshold = confidence_threshold
        self.hold_frames = hold_frames
        self.reset()

    def update(self, letter, conf):
        if letter is None or conf < self.confidence_threshold:
            self.reset()
            return None

        if letter != self.curr_letter:
            self.curr_letter = letter
            self.letter_frames = 1
            self.locked = False
        else:
            self.letter_frames += 1

        if self.letter_frames >= self.hold_frames and not self.locked:
            self.locked = True
            return letter

        return None

    @property
    def progress(self):
        # progress to letter being typed
        return min(self.letter_frames / self.hold_frames, 1.0)

    def reset(self):
        self.curr_letter = None
        self.letter_frames = 0
        self.locked = False
        