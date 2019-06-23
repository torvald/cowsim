#!/usr/bin/python3
import multiprocessing
import random
import time
import math
import numpy
from pprint import pprint

debug = True
tests = True
threads = 1 if debug else multiprocessing.cpu_count()-2

sec_per_step = 20
need_water_per_24hour = 100
use_water_per_step = need_water_per_24hour / 24 / 60 / 60 * sec_per_step
need_grass_per_24hour = 30
use_grass_per_step = need_grass_per_24hour / 24 / 60 / 60 * sec_per_step
drinks_per_step = 1
eats_grass_per_step = 0.5
eats_consentrate_per_step = 0.3


config = {'steps': 50000, 'barn_width': 20,
        'barn_height' : 20,
        'number_of_cows' : 10,
        'generate_stats': False,
        'max_water': 100,
        'max_grass': 30,
        'max_concentrate': 20,
        'max_milk': 30,
        }

class Simulation:
    config = None
    barn = None
    cows = None
    step = 0

    def __init__(self, config):
        self.barn = Barn(config)
        self.config = config

        self.cows = [Cow(self.barn) for x in range(config['number_of_cows'])]

        walls = [(6, 9, 2, 4)]

        for wall in walls:
            x, y, direction, length = wall
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
            self.barn.place_agent(cow, (x, y)) 

        f = Feeder(self) 
        x, y = None, None
        while True:
            x = random.randrange(config['barn_width'])
            y = random.randrange(config['barn_height'])
            if self.barn.is_cell_empty((x, y)): break
        self.barn.place_agent(f, (x, y)) 

        w = Water(self) 
        x, y = None, None
        while True:
            x = random.randrange(config['barn_width'])
            y = random.randrange(config['barn_height'])
            if self.barn.is_cell_empty((x, y)): break
        self.barn.place_agent(w, (x, y)) 

    def state(self):
        return {'barn': self.barn.state(),
                'step': self.step}

    def http_rep(self, state):
        output = "Step: {}\n".format(state['step'])
        grid = state['barn']
        for x in grid:
            for y in x:
                if y:
                    output += str(max(y, key=lambda agent: agent.weight))
                else:
                    output += " "

            output += "\n"
        return output

    def run(self):
        for s in range(self.config['steps']):
            self.step = s
            for cow in self.cows:
                if cow.alive: cow.step()
            if s % 100 == 0 and self.config['generate_stats']:
                file = open("stats.json","w") 
                file.write(self.http_rep(self.state())) 
                file.close() 


class Barn():
    grid = None
    config = None

    def __init__(self, config):
        self.config = config
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
    alive = True
    moving = False

    current_target = None
    current_path = None

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
    grass = None
    water = None
    concentrates = None
    milk = None
    sleep = None

    current_objective = None

    def __init__(self, model):
        super().__init__(model)
        self.water = random.randrange(model.config['max_water'])
        self.grass = random.randrange(model.config['max_grass'])
        self.concentrates = random.randrange(model.config['max_concentrate'])
        self.milk = random.randrange(model.config['max_milk'])

    def step(self):
        self._update_healt()
        self._update_target()
        self.random_move()

    def _update_healt(self):
        # Assuming on step is about 15 sec
        pass

    def _update_target(self):
        new_objective = self._calc_objective()
        if new_objective == self.current_objective:
            return self.current_target

    def _calc_objective(self):
        # if on the move, easier to change cow's mind
        if self.moving:
            if self.water < 50: return "drink"
            if self.concentrates < 10: return "eat_concentrates"
            if self.grass < 10: return "eat_grass"
        # doing somthing, chaning my mind takes more effort
        else:
            if self.water < 25: return "drink"
            if self.concentrates < 5: return "eat_concentrates"
            if self.grass < 5: return "eat_grass"
        return "sleep"


    def healt(self):
        pass
        #if self.water <= 0 or selg.grass <= 0:

    def __repr__(self):
        if not self.alive: return 'X'
        return 'K'

class Wall(Agent):
    def __init__(self, model):
        super().__init__(model)

    def __repr__(self):
        return '#'

class Bed(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def __repr__(self):
        return 'B'

class Feeder(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def __repr__(self):
        return 'F'

class Water(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def __repr__(self):
        return 'W'

class Grass(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def __repr__(self):
        return 'G'

class OneWayGate(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)
        self.direction = direction

    def valid_entry_grid(self, pos):
        gate_pos_x, gate_pos_y = self.pos
        entry_pos_x, entry_pos_y = pos
        if self.direction == "→":
            if gate_pos_x == entry_pos_x-1 and gate_pos_y == entry_pos_y: return True
        if self.direction == "←":
            if gate_pos_x == entry_pos_x+1 and gate_pos_y == entry_pos_y: return True
        if self.direction == "↑":
            if gate_pos_x == entry_pos_x and gate_pos_y == entry_pos_y-1: return True
        if self.direction == "↓":
            if gate_pos_x == entry_pos_x and gate_pos_y == entry_pos_y+1: return True
        return False

    def __repr__(self):
        return self.direction

class SmartGate(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)
        self.direction = direction

    def __repr__(self):
        return '#'

def sim(config):
    sim = Simulation(config)
    sim.run()

    rand = bool(random.getrandbits(1))
    return rand

from http.server import HTTPServer, BaseHTTPRequestHandler

class web_server(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/stats.json'
        try:
            #Reading the file
            file_to_open = open(self.path[1:]).read()
            self.send_response(200)
        except:
            file_to_open = "File not found"
            self.send_response(404)
        self.end_headers()
        self.wfile.write(bytes(file_to_open, 'utf-8'))

def http_server():
    httpd = HTTPServer(('193.35.52.75', 8585), web_server)
    httpd.serve_forever()

if __name__ == '__main__':
    print ("There are ", threads, " threads")

    start = time.time()

    jobs = []
    for k in range(threads):
        jobs.append(config)

    jobs[0]['generate_stats'] = True

    http_server = multiprocessing.Process(name='http_server', target=http_server)
    pool = multiprocessing.Pool(threads)

    http_server.start()
    results = pool.map(sim, jobs)
    print(results)

    end = time.time()

    print ("The average number of steps is ", int(sum(results) / len(results)), " steps")
    print ("The simulation(s) took ", end - start, " seconds to run")

print("Mean:", numpy.mean(results), " Median:", numpy.median(results), " Stdev:", numpy.std(results))
