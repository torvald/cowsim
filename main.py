#!/usr/bin/python3
import multiprocessing
import random
import time
import math
import numpy
from pprint import pprint

debug = True
threads = 1 if debug else multiprocessing.cpu_count()

config = {'barn_width': 10,
              'barn_height' : 10,
              'number_of_cows' : 5}

class Simulation:
    config = None
    barn = None
    cows = None

    def __init__(self, config):
        self.barn = Barn(config)
        self.cows = Cows(config)
        self.config = config

        walls = [(6, 9, 2, 4)]

        for wall in walls:
            x, y, direction, length = wall
           # y = wall[1]
           # direction = wall[2]
           # length = wall[3]
            w = Wall()
            self.barn.place(w, (x,y))
            for n in range(length):
                if direction == 6:
                    x += 1
                elif direction == 2:
                    y -= 1
                elif direction == 4:
                    x -= 1
                elif direction == 8:
                    y += 1
                w = Wall()
                self.barn.place(w, (x,y))

        for i in range(self.cows.count()):
            x, y = None, None
            while True:
                x = random.randrange(config['barn_width'])
                y = random.randrange(config['barn_height'])
                if self.barn.is_cell_empty((x, y)): break
#            energy = self.random.randrange(2 * self.cows_gain_from_food)
#            cows = Cow(self.next_id(), (x, y), self, False, energy)
            cow = Cow()
            self.barn.place(cow, (x, y)) 
        #    self.schedule.add(cows)


    def state(self):
        return {'barn': self.barn.state()}

    def run(self):
        pass

class Cows:
    cows = None

    def __init__(self, config):
        self.cows = [Cow() for x in range(config['number_of_cows'])]

    def count(self):
        return len(self.cows)

    def cows(self):
        return self.cows

class Barn:
    barn = None

    def __init__(self, config):
        w, h = config['barn_width'], config['barn_height']
        self.barn = [[0 for x in range(w)] for y in range(h)]

    def is_cell_empty(self, x_y):
        x, y = x_y
        return False if self.barn[x][y] else True

    def place(self, cow, x_y):
        x, y = x_y
        # TODO: check
        self.barn[x][y] = cow

    def state(self):
        return self.barn

class Wall:
    def __repr__(self):
        return '#'

class Cow:
    water = 50

    def water(self):
        return water

    def __repr__(self):
        return 'K'

def sim(config):
    sim = Simulation(config)
    pprint(sim.state())
    rand = bool(random.getrandbits(1))

    return rand


if __name__ == '__main__':
    print ("There are ", threads, " threads")

    start = time.time()

    jobs = []
    for k in range(threads):
        jobs.append(config)

    pool = multiprocessing.Pool(threads)

    results = pool.map(sim, jobs)
    print(results)

    end = time.time()
    
    print ("The average number of steps is ", int(sum(results) / len(results)), " steps")
    print ("The simulation(s) took ", end - start, " seconds to run")

print("Mean:", numpy.mean(results), " Median:", numpy.median(results), " Stdev:", numpy.std(results))
