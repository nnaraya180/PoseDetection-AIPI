import pygame
import cv2 as _cv2
import numpy as np

pygame.init()
_screen = None

def imshow(title, frame):
    global _screen
    h, w = frame.shape[:2]
    if _screen is None:
        _screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(title)
    rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))
    _screen.blit(surface, (0,0))
    pygame.display.flip()

def waitKey(delay):
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        return ord('q')
    return -1

def destroyAllWindows():
    pygame.quit()
