import pygame
from pygame import *

def main():
    pygame.init()
    screen = pygame.display.set_mode((400,400))

    color_dict = {'black': [0, 0.25], 'red':[0.25, 0.5], 'orange': [0.5, 0.75], 'green': [0.75, 1]}

    continued = 1

    while continued:

        for event in pygame.event.get():
            if event.type == QUIT:
                continued = 0
            elif event.type == KEYDOWN:
                handle_keys(event.key, player)

        pygame.display.flip()

if __name__ == '__main__':
    main()