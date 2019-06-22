#!/usr/bin/python3
import multiprocessing
import random
import time
import math
import numpy
from pprint import pprint

debug = True
tests = True
threads = 1 if debug else multiprocessing.cpu_count()

config = {'steps': 5, 'barn_width': 50,
              'barn_height' : 50,
              'number_of_cows' : 10}

class Simulation:
    config = None
    barn = None
    cows = None

    def __init__(self, config):
        self.barn = Barn(config)
        self.config = config

        self.cows = [Cow(self.barn) for x in range(config['number_of_cows'])]

        walls = [(6, 9, 2, 4)]

        for wall in walls:
            x, y, direction, length = wall
           # y = wall[1]
           # direction = wall[2]
           # length = wall[3]
            w = Wall(self.barn)
            self.barn.place_agent(w, (x,y))
            for n in range(length):
                if direction == 6:
                    x += 1
                elif direction == 2:
                    y -= 1
                elif direction == 4:
                    x -= 1
                elif direction == 8:
                    y += 1
                w = Wall(self.barn)
                self.barn.place_agent(w, (x,y))

        for cow in self.cows:
            x, y = None, None
            while True:
                x = random.randrange(config['barn_width'])
                y = random.randrange(config['barn_height'])
                if self.barn.is_cell_empty((x, y)): break
#            energy = self.random.randrange(2 * self.cows_gain_from_food)
#            cows = Cow(self.next_id(), (x, y), self, False, energy)
            self.barn.place_agent(cow, (x, y)) 
        #    self.schedule.add(cows)


    def state(self):
        return {'barn': self.barn.state()}

    def run(self):
        for s in range(self.config['steps']):
            for cow in self.cows:
                cow.step()


#class Cows:
#    cows = None
#
#    def __init__(self, cows):
#        self.cows = cows
#
#    def count(self):
#        return len(self.cows)
#
#    def cows(self):
#        return self.cows

class Barn():
    grid = None

    def __init__(self, config):
        w, h = config['barn_width'], config['barn_height']
        self.grid = [[[] for x in range(w)] for y in range(h)]

    def neighborhood(self, pos):
        x, y = pos
        neighborhood = []
        for x_delta, y_delta in [(1,0),(-1,0),(0,1),(0,-1)]:
            if self.valid_cell((x+x_delta,y+y_delta)):
                neighborhood.append((x+x_delta,y+y_delta))
        return neighborhood

    def valid_cell(self, pos):
        x, y = pos
        if x < 0 or x >= len(self.grid): return False
        if y < 0 or y >= len(self.grid[0]): return False
        return True

    def is_cell_walkable(self, pos):
        if not self.valid_cell: return False
        x, y = pos
        for obj in self.grid[x][y]:
            if isinstance(obj, Cow): return False
            if isinstance(obj, Wall): return False
        return True

    def is_cell_empty(self, pos):
        x, y = pos
        return True if len(self.grid[x][y]) == 0 else False

    def move_agent(self, agent, new_pos):
        if tests:
            assert(self.is_cell_walkable(new_pos))
        old_x, old_y = agent.pos
        new_x, new_y = new_pos
        self.grid[old_x][old_y].remove(agent)
        self.grid[new_x][new_y].append(agent)
        agent.update_pos(new_pos)

    def place_agent(self, cow, pos):
        x, y = pos
        # TODO: check
        self.grid[x][y].append(cow)
        cow.update_pos(pos)

    def state(self):
        return self.grid

class Agent(object):
    model = None
    weight = None
    pos = None

    def __init__(self, model, weight=100):
        self.model = model
        self.weight = weight

    def update_weight(self, weight):
        self.weight = weight

    def update_pos(self,pos):
        self.pos = pos


class WalkingAgent(Agent):

    def __init__(self, model):
        # Ensure Walking Agents always stay on top of other objects
        super().__init__(model, weight=101)

    def random_move(self):
        next_moves = [pos for pos in self.model.neighborhood(self.pos) if self.model.is_cell_walkable(pos)]
        if len(next_moves) > 0:
            next_move = random.choice(next_moves)
            self.model.move_agent(self, next_move)
        else:
            print("No legal moves for cow in {}".format(self.pos))

class Cow(WalkingAgent):
    water = 50

    def __init__(self, model):
        super().__init__(model)

    def step(self):
        self.random_move()

    def water(self):
        return water

    def __repr__(self):
        return 'K'

class Wall(Agent):
    def __init__(self, model):
        super().__init__(model)

    def __repr__(self):
        return '#'

def sim(config):
    sim = Simulation(config)
    #pprint(sim.state())
    sim.run()
    #pprint(sim.state())

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
