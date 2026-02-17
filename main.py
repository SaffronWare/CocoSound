import numpy as np
import pygame as pg

COORDINATE_SYSTEM = (1,1)

NUM_GRIDS_X = 400
NUM_GRIDS_Y = 400

DELX = COORDINATE_SYSTEM[0]*2 / (NUM_GRIDS_X)
DELY = COORDINATE_SYSTEM[1]*2 / (NUM_GRIDS_Y)

FPS = 60
DT = 1/FPS

xcoordinatesraw = np.linspace(-COORDINATE_SYSTEM[0], COORDINATE_SYSTEM[0], NUM_GRIDS_X,endpoint=False)
ycoordinatesraw = np.linspace(-COORDINATE_SYSTEM[1],COORDINATE_SYSTEM[1],NUM_GRIDS_Y,endpoint=False)
xcoordinates, ycoordinates =  np.meshgrid(xcoordinatesraw,ycoordinatesraw)
pressures = np.zeros((NUM_GRIDS_X,NUM_GRIDS_Y), dtype=np.float32)
gradients = np.zeros((NUM_GRIDS_X,NUM_GRIDS_Y), dtype=np.float32)
soundspeed = np.full((NUM_GRIDS_X,NUM_GRIDS_Y),343,dtype=np.float32)


def perturb(intensity, position=(0,0), radius=0.1):
    mask = (xcoordinates**2 + ycoordinates**2) <= radius**2

    pressures[mask] += intensity

def pressure_color():
    image = np.zeros((NUM_GRIDS_X,NUM_GRIDS_Y,3))

    mask = pressures >= 0
    red = 255 - 255*np.exp(-pressures)
    blue = 255 - 255 *np.exp(pressures)

    image[...,0] = np.where(mask, red, 0)
    image[...,2] = np.where(~mask, blue, 0)

    return image

def main():
    pg.init()
    clock = pg.time.Clock()

    window_dimensions = (NUM_GRIDS_X,NUM_GRIDS_Y)
    window = pg.display.set_mode(window_dimensions)

    perturb(1)

    running = True
    while running:
        surf_sound = pg.surfarray.make_surface(pressure_color())

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False 

        window.blit(surf_sound, (0,0))

        clock.tick(FPS)
        pg.display.flip()
    
    pg.quit()

if __name__ == '__main__':
    main()