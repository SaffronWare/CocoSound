import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
from math import cos

COORDINATE_SYSTEM = (20,20)
NUM_GRIDS_X = 600
NUM_GRIDS_Y = 600
DELX = COORDINATE_SYSTEM[0]*2 / (NUM_GRIDS_X)
DELY = COORDINATE_SYSTEM[1]*2 / (NUM_GRIDS_Y)
FPS =300
dt= 1/FPS
SPEED=0.01
DT = dt*SPEED
C = 500

xcoordinatesraw = np.linspace(-COORDINATE_SYSTEM[0], COORDINATE_SYSTEM[0], NUM_GRIDS_X,endpoint=False,dtype=np.float32)
ycoordinatesraw = np.linspace(-COORDINATE_SYSTEM[1],COORDINATE_SYSTEM[1],NUM_GRIDS_Y,endpoint=False,dtype=np.float32)
xcoordinates, ycoordinates =  np.meshgrid(xcoordinatesraw,ycoordinatesraw)
pressures = np.zeros((NUM_GRIDS_X,NUM_GRIDS_Y), dtype=np.float32)
gradients = np.zeros((NUM_GRIDS_X,NUM_GRIDS_Y), dtype=np.float32)
soundspeed = np.full((NUM_GRIDS_X,NUM_GRIDS_Y),C,dtype=np.float32)


def sp(x,y):
    x2 = 0.5 * (x+COORDINATE_SYSTEM[0])/COORDINATE_SYSTEM[0] *  NUM_GRIDS_X
    
    y2 = 0.5 * (y+COORDINATE_SYSTEM[1])/COORDINATE_SYSTEM[1] *  NUM_GRIDS_Y
    return (x2,y2)

def ps(x,y):
    x2 = 2*x/NUM_GRIDS_X * COORDINATE_SYSTEM[0] - COORDINATE_SYSTEM[0]
    y2 = 2*y/NUM_GRIDS_Y * COORDINATE_SYSTEM[1] - COORDINATE_SYSTEM[1]
    return (x2,y2)

def measure_at(x,y):
    
    mask  = (np.less_equal(np.abs(x - xcoordinates), DELX)) & (np.less_equal(np.abs(y-ycoordinates), DELY))
    return pressures[mask][0]
def set_boundaries():
    mask = np.abs(np.abs(xcoordinates + ycoordinates) + np.abs(ycoordinates - xcoordinates)-30) < 5

    soundspeed[mask]=250
    #pressures[mask] =50


def perturb(intensity, radius=0.01, x=0,y=0):
    global pressures
    mask = ((xcoordinates-x)**2 + (ycoordinates-y)**2) <= radius**2
    pressures[mask] = intensity

def pressure_color():
    image = np.zeros((NUM_GRIDS_X,NUM_GRIDS_Y,3))

    mask = (pressures >= 0)
    red = 255 - 255*np.exp(-pressures)
    blue = 255 - 255 *np.exp(pressures)


    mask2 = ( soundspeed < C*0.9)& (np.abs(pressures) < 0.1)
    #image[...,0] = np.where(mask2, 255,0)
    image[...,1] = np.where(mask2, 20,0)
    #image[...,2] = np.where(mask2, 255,0)

    
    image[...,0] = np.where(mask, red, 0)
    image[...,2] = np.where(~mask, blue, 0)


    return image

def update_gradients():
    laplacianx = pressures[:,:-2] - 2*pressures[:, 1:-1] + pressures[:, 2:]
    laplacianx /= DELX**2
    laplaciany = pressures[:-2, :] - 2*pressures[1:-1, :] + pressures[2:, :]
    laplaciany /= DELY**2
    gradients[:,1:-1] += (laplacianx)*DT*(soundspeed[:,1:-1])**2
    gradients[1:-1,:] += laplaciany * DT * soundspeed[1:-1,:]**2

def update_pressures():
    global pressures
    pressures += gradients * DT

def compute_fourrier(fourrier,time):
    res = 0
    for f in fourrier:
        res+=2* cos(f[1] * time * 2 * np.pi)  *f[0]

    return res
def apply_reflective_boundary():
    # left & right
    pressures[0, :]  = pressures[1, :]
    pressures[-1, :] = pressures[-2, :]

    # top & bottom
    pressures[:, 0]  = pressures[:, 1]
    pressures[:, -1] = pressures[:, -2]

def main():
    pg.init()
    clock = pg.time.Clock()

    
    showing = True
    
    window_dimensions = (NUM_GRIDS_X,NUM_GRIDS_Y)
    window = None 
    if showing:
        window = pg.display.set_mode(window_dimensions)


    sources = [(4,-15), (-4,-8), (10,7), (-5,3)]
    fourrier = [[(2,600)], [(1.67,670)], [(3, 420)], [(1,1000)]]

    #perturb(10,0.05)
    time = 0
    frequency = 600 # times per seocnd
    interval = 1/frequency
    print(interval)
    running = True
    set_boundaries()
    microphones = [(0,10),(0,0),(0,-15)]

    times = []
    measurements = [[] for i in range(len(microphones))]

    pg.font.init()
    deflt = pg.font.Font(None,20)
    time_start = 0
    
    if showing:
        while running and time <= 0.1:
            window.fill((0,0,0))
            time += DT
            timetxt = deflt.render(f"time : {time}", True, (255,255,255))
            #timetxt = pg.font.(f"time : {time}")
            #print(time)
            for s,f in zip(sources,fourrier):
                perturb(compute_fourrier(f,time),0.1,s[0],s[1])
            
        
            

            #perturb(0.1)
            
            surf_sound = pg.surfarray.make_surface(pressure_color())
            
            #perturb(0.1)
            update_gradients()
            update_pressures()
            apply_reflective_boundary()
            
            times.append(time)
            for i,point_to_measure in enumerate(microphones):
                measurements[i].append(measure_at(point_to_measure[0],point_to_measure[1]))

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False 
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_r:
                        time_start = time

            window.blit(surf_sound, (0,0))
            for point_to_measure in microphones:
                pg.draw.circle(window,(0,200,0), sp(point_to_measure[0],point_to_measure[1]),5, 1)
            window.blit(timetxt, (0,0))


            

            clock.tick(FPS)
            pg.display.flip()
    else:
        i=0
        while time < 0.05:
            pass

    pg.quit()



    for m in measurements:
        plt.plot(times,m)
    plt.show()

    

if __name__ == '__main__':
    main()