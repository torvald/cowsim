#!/usr/bin/python3
import multiprocessing
import random
import time
import math
import numpy
import collections
from pprint import pprint, pformat

debug = True
tests = True
slowmo = True
threads = 1 if debug else multiprocessing.cpu_count()-2

sec_per_step = 20
need_water_per_24hour = 100
use_water_per_step = need_water_per_24hour / 24.0 / 60 / 60 * sec_per_step
need_grass_per_24hour = 30
use_grass_per_step = need_grass_per_24hour / 24.0 / 60 / 60 * sec_per_step
drinks_per_step = 1
eats_grass_per_step = 0.5
eats_consentrate_per_step = 0.3


config = {'steps': 50000, 'barn_height': 20,
        'barn_width' : 70,
        'number_of_cows' : 20,
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

        self.cows = [Cow(self) for x in range(config['number_of_cows'])]
        if debug: self.cows[0].debug(True)

        walls = []
        for i in range(10):
            x,y = self.random_pos()
            direction = random.choice([2,4,6,8])
            length = random.randrange(10)
            walls.append((x,y,direction,length))

        for wall in walls:
            x, y, direction, length = wall
            w = Wall(self)
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
                w = Wall(self)
                if self.barn.valid_cell((x,y)):
                    self.barn.place_agent(w, (x,y))

        for cow in self.cows:
            x, y = None, None
            while True:
                x, y = self.random_pos()
                if self.barn.is_cell_empty((x, y)): break
            self.barn.place_agent(cow, (x, y))

        for i in range(10):
            g = Grass(self)
            x, y = None, None
            while True:
                x, y = self.random_pos()
                if self.barn.is_cell_empty((x, y)): break
            self.barn.grass_positions.append((x,y))
            self.barn.place_agent(g, (x, y))

        for i in range(5):
            f = Feeder(self)
            x, y = None, None
            while True:
                x, y = self.random_pos()
                if self.barn.is_cell_empty((x, y)): break
            self.barn.concentrate_positions.append((x,y))
            self.barn.place_agent(f, (x, y))

        for i in range(10):
            w = Water(self)
            x, y = None, None
            while True:
                x, y = self.random_pos()
                if self.barn.is_cell_empty((x, y)): break
            self.barn.water_positions.append((x,y))
            self.barn.place_agent(w, (x, y))

        for i in range(20):
            b = Bed(self)
            x, y = None, None
            while True:
                x, y = self.random_pos()
                if self.barn.is_cell_empty((x, y)): break
            self.barn.sleep_positions.append((x,y))
            self.barn.place_agent(b, (x, y)) 

    def random_pos(self):
        x = random.randrange(config['barn_height'])
        y = random.randrange(config['barn_width'])
        return (x,y)

    def state(self):
        debug_cow = next(cow for cow in self.cows if cow.debug)
        dead = len([cow for cow in self.cows if not cow.alive])
        water = [cow.water for cow in self.cows]
#print("Mean:", numpy.mean(results), " Median:", numpy.median(results), " Stdev:", numpy.std(results))
        return {'model': {'dead': dead, 'water_mean': numpy.mean(water),
            'water_median': numpy.median(water), 'water_stdev': numpy.std(water)},
                'barn': self.barn.state(),
                'step': self.step,
                'debug_cow': debug_cow}

    def http_rep(self, state):
        output = "Step: {}\n".format(state['step'])
        grid = state['barn']['grid']
        debug_cow_path = state['debug_cow'].current_path
        debug_cow_state = state['debug_cow'].state()
        model = state['model']
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[x][y]:
                    output += str(max(grid[x][y], key=lambda agent: agent.weight))
                elif (x,y) in debug_cow_path:
                    output += "o"
                else:
                    output += " "

            output += "\n"
        output += pformat(model, indent=4)
        output += "\n"
        output += pformat(debug_cow_state, indent=4)

        return output

    def run(self):
        for s in range(self.config['steps']):
            self.step = s
            for cow in self.cows:
                if cow.alive: cow.step()

            stats_modulo = 100
            if slowmo:
                stats_modulo = 1
                time.sleep(0.1)

            if s % stats_modulo == 0 and self.config['generate_stats']:
                file = open("stats.json","w") 
                file.write(self.http_rep(self.state())) 
                file.close() 


class Barn():
    grid = None
    grass_positions = []
    water_positions = []
    concentrate_positions = []
    sleep_positions = []

    def __init__(self, config):
        w, h = config['barn_height'], config['barn_width']
        self.grid = [[[] for x in range(h)] for y in range(w)]

    def neighborhood(self, pos):
        x, y = pos
        neighborhood = []
        for x_delta, y_delta in [(1,0),(-1,0),(0,1),(0,-1)]:
            if self.valid_cell((x+x_delta,y+y_delta)):
                neighborhood.append((x+x_delta,y+y_delta))
        return neighborhood

    def neighbors(self, pos, include_center=False):
        x,y = pos
        neighbors = [self.grid[x][y]] if include_center else []
        for pos in self.neighborhood(pos):
            x,y = pos
            neighbors += self.grid[x][y]
        return neighbors

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
            x,y = new_pos
            assert(self.is_cell_walkable(new_pos))
        old_x, old_y = agent.pos
        new_x, new_y = new_pos
        self.grid[old_x][old_y].remove(agent)
        self.grid[new_x][new_y].append(agent)
        agent.update_pos(new_pos)

    def place_agent(self, agent, pos):
        x, y = pos
        # TODO: check
        self.grid[x][y].append(agent)
        agent.update_pos(pos)

    def state(self):
        return {'grid': self.grid}

class Agent(object):
    model = None
    weight = None
    pos = None
    debug = False

    def __init__(self, model, weight=100):
        self.model = model
        self.weight = weight

    def update_weight(self, weight):
        self.weight = weight

    def update_pos(self,pos):
        self.pos = pos

    def debug(self, value):
        self.debug = value


class WalkingAgent(Agent):
    alive = True
    moving = False

    current_target = None
    current_path = None

    def __init__(self, model):
        # Ensure Walking Agents always stay on top of other objects
        super().__init__(model, weight=101)

    def random_move(self):
        next_moves = [pos for pos in self.model.barn.neighborhood(self.pos) if self.model.barn.is_cell_walkable(pos)]
        if len(next_moves) > 0:
            next_move = random.choice(next_moves)
            self.model.barn.move_agent(self, next_move)
        else:
            print("No legal moves for cow in {}".format(self.pos))
            print(self.model.barn.neighborhood(self.pos))
            time.sleep(10)

    def move(self):
        # [0] is start posistion, [1] is where we want to go next
        target = self.current_target

        if len(self.current_path) == 0:
            print ("bfs thinks you should stay put") 
            #self.current_path = None
            return

        if len(self.current_path) == 1:
            # good? [0] is where we are at, sit still
            #self.current_path = None
            return

        self.current_path = self.current_path[1:]
        next_move = self.current_path[0]
        if self.model.barn.is_cell_walkable(next_move):
            self.model.barn.move_agent(self, next_move)
        else:
            print("{} is not walkable, forcing new search".format(next_move))
            self.current_objective = None
            #self.random_move()

    def bfs(self, grid, start, target):
        queue = collections.deque([[start]])
        seen = set([start])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x,y) == target:
                return path
            explore = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            random.shuffle(explore)
            for x2, y2 in explore:
                # ignore cells out of bound
                if not self.model.barn.valid_cell((x2, y2)): continue
                # Dont search though walls
                agents = grid[x2][y2]
                if any(type(agent) is Wall for agent in agents): continue
                # and dont goto cells we have seen before
                if (x2, y2) in seen: continue
                # if we search through a onewaygate, check that you are on valid side
                if any(type(agent) is OneWayGate for agent in agents):
                    on_way_gate = list(filter(lambda x: type(x) is OneWayGate, agents))[0]
                    if on_way_gate.valid_entry_grid((x,y)): continue
                # looks good, lets search further
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
        return []

class Cow(WalkingAgent):
    grass = None
    water = None
    concentrates = None
    milk = None
    sleep = None

    current_objective = None

    def __init__(self, model):
        super().__init__(model)
        self.water = random.randrange(model.config['max_water']/2, model.config['max_water'])
        self.grass = random.randrange(model.config['max_grass']/2, model.config['max_grass'])
        self.concentrates = random.randrange(model.config['max_concentrate']/2, model.config['max_concentrate'])
        self.milk = random.randrange(model.config['max_milk']/2,model.config['max_milk'])

    def step(self):
        self._update_state()
        new_objective = self._calc_objective()
        if new_objective != self.current_objective:
            self._update_target(new_objective)
        # move toward current target
        if tests:
            assert self.current_objective is not None
            assert self.current_target is not None
        self.move() 

    def _update_state(self):
        self.water -= use_water_per_step
        self.grass -= use_grass_per_step
        neighbors = self.model.barn.neighbors(self.pos, include_center=True)

        if any(type(agent) is Water for agent in neighbors) and self.water < self.model.config['max_water']: self.water += drinks_per_step
        if any(type(agent) is Grass for agent in neighbors) and self.grass < self.model.config['max_grass']: self.grass += eats_grass_per_step
        if any(type(agent) is Feeder for agent in neighbors) and self.concentrates < self.model.config['max_concentrate']: self.concentrates += eats_consentrate_per_step

        if self.water <= 0 or self.grass <= 0:
            self.alive = False

    def _update_target(self, new_objective):
        self.current_objective = new_objective
        if new_objective == "eat_grass":
            self.current_target = random.choice(self.model.barn.grass_positions)
        if new_objective == "drink":
            self.current_target = random.choice(self.model.barn.water_positions)
        if new_objective == "eat_concentrates":
            self.current_target = random.choice(self.model.barn.concentrate_positions)
        if new_objective == "sleep":
            self.current_target = random.choice(self.model.barn.sleep_positions)
        self.current_path = self.bfs(self.model.barn.grid, self.pos, self.current_target)

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

    def state(self):
        return {'pos': self.pos,
                'alive': self.alive,
                'current_target': self.current_target,
                'current_objective': self.current_objective,
                'water': self.water,
                'concentrates': self.concentrates,
                'grass': self.grass,
                'path': self.current_path,
                }

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
