import numpy as np
import cv2

# Class that can take the output of a stable fluids frame and render
# it into a color buffer.
class Renderer():

    def __init__(self, w, h):
        self.w = w
        self.h = h
        pass

    def render(self, input):
        # Simple reinhard "tone mapping" to scale values between 0 and 1.
        img = np.clip(input, 0, 1)
        img = np.array(255 * np.repeat(img[:,:,np.newaxis], 3, axis=2),
            dtype=np.uint8)
        resized = cv2.resize(img, dsize=(self.w, self.h),
            interpolation=cv2.INTER_LINEAR)
        return resized
