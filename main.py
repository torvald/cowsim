#!/usr/bin/python3
import multiprocessing
import random
import time
import math
import numpy
import collections
import json
from pprint import pprint, pformat

debug = True
tests = True
slowmo = False
profile = False
heatmap = True
threads = 1 if debug else multiprocessing.cpu_count() - 2

sec_per_step = 20
need_water_per_24hour = 100
use_water_per_step = need_water_per_24hour / 24.0 / 60 / 60 * sec_per_step
need_grass_per_24hour = 30
need_consentrate_per_24hour = 15
use_grass_per_step = need_grass_per_24hour / 24.0 / 60 / 60 * sec_per_step
use_consentrate_per_step = need_consentrate_per_24hour / 24.0 / 60 / 60 * sec_per_step
drinks_per_step = 1
eats_grass_per_step = 0.5
eats_consentrate_per_step = 0.3

directions = {2: "↓", 4: "←", 6: "→", 8: "↑"}

config = {
    "steps": 10000,
    "barn_height": 20,
    "barn_width": 70,
    "number_of_cows": 40,
    "generate_stats": False,
    "max_water": 100,
    "max_grass": 30,
    "max_concentrate": 20,
    "max_milk": 30,
}


class Simulation:
    config = None
    barn = None
    cows = None
    step = 0

    def __init__(self, config):
        self.barn = Barn(config)
        self.config = config

        self.cows = [Cow(self) for x in range(config["number_of_cows"])]
        if debug:
            self.cows[0].update_debug(True)

        def place_random_agents(
            agent_type, groups, max_agents, min_agents=1, cluster=False
        ):
            assert min_agents <= max_agents
            agents = []
            for i in range(groups):
                x, y = self.random_pos()
                direction = random.choice([2, 4, 6, 8])
                number_of_agents = random.randrange(min_agents, max_agents + 1)
                agents.append((x, y, direction, number_of_agents))
            for agent in agents:
                x, y, direction, number_of_agents = agent
                posistions = [(x, y)]
                for n in range(number_of_agents - 1):
                    direction = random.choice([2, 4, 6, 8]) if cluster else direction
                    if direction == 6:
                        x += 1
                    elif direction == 8:
                        y -= 1
                    elif direction == 4:
                        x -= 1
                    elif direction == 2:
                        y += 1
                    if self.barn.valid_cell((x, y)) and self.barn.is_cell_empty((x, y)):
                        posistions.append((x, y))
                # convert to set, to get uniqe posistions, clusters may generate the same pos many times
                for x, y in set(posistions):
                    if agent_type == "wall":
                        self.barn.place_agent(Wall(self), (x, y))
                    if agent_type == "onewaygate":
                        self.barn.place_agent(OneWayGate(self, direction), (x, y))
                    if agent_type == "grass":
                        self.barn.place_agent(Grass(self), (x, y))
                        self.barn.grass_positions.append((x, y))
                    if agent_type == "water":
                        self.barn.place_agent(Water(self), (x, y))
                        self.barn.water_positions.append((x, y))
                    if agent_type == "feeder":
                        self.barn.place_agent(Feeder(self), (x, y))
                        self.barn.concentrate_positions.append((x, y))
                    if agent_type == "bed":
                        self.barn.place_agent(Bed(self), (x, y))
                        self.barn.sleep_positions.append((x, y))

        place_random_agents("wall", groups=50, max_agents=10, min_agents=3)
        place_random_agents("onewaygate", groups=0, max_agents=1)
        place_random_agents("grass", groups=5, max_agents=5)
        place_random_agents("water", groups=5, max_agents=3)
        place_random_agents("feeder", groups=5, max_agents=2)
        place_random_agents("bed", groups=5, max_agents=20, min_agents=20, cluster=True)

        for cow in self.cows:
            x, y = None, None
            while True:
                x, y = self.random_pos()
                if self.barn.is_cell_empty((x, y)):
                    break
            self.barn.place_agent(cow, (x, y))

    def random_pos(self):
        x = random.randrange(config["barn_height"])
        y = random.randrange(config["barn_width"])
        if self.barn.is_cell_empty((x, y)):
            return (x, y)
        return self.random_pos()

    def state(self):
        debug_cow = next(cow for cow in self.cows if cow.debug)
        dead = len([cow for cow in self.cows if not cow.alive])
        water = [cow.water for cow in self.cows if cow.alive]
        stuck_recalc = [cow.stuck_recalc for cow in self.cows if cow.alive]
        return {
            "model": {
                "dead": dead,
                "water_mean": numpy.mean(water),
                "water_median": numpy.median(water),
                "water_stdev": numpy.std(water),
                "stuck_recalc_mean": numpy.mean(stuck_recalc),
                "stuck_recalc_median": numpy.median(stuck_recalc),
                "stuck_recalc_stdev": numpy.std(stuck_recalc),
                "step": self.step,
            },
            "debug_cow": debug_cow,
            "barn": self.barn.state(),
        }

    def human_readable_state(self, state):
        output = "Step: {}\n".format(state["model"]["step"])
        grid = state["barn"]["grid"]
        debug_cow_path = state["debug_cow"].current_path
        debug_cow_state = state["debug_cow"].state()
        model = state["model"]
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[x][y]:
                    output += max(
                        grid[x][y], key=lambda agent: agent.weight
                    ).ASCIIDraw()
                elif (x, y) in debug_cow_path:
                    output += "*"
                else:
                    output += " "

            output += "\n"
        output += pformat(model, indent=4)
        output += "\n"
        output += pformat(debug_cow_state, indent=4)

        return output

    def json_state(self, state):
        grid = state["barn"]["grid"]
        debug_cow_path = state["debug_cow"].current_path
        debug_cow_state = state["debug_cow"].state()
        w, h = len(grid), len(grid[0])
        grid_json = [[[] for x in range(h)] for y in range(w)]
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[x][y]:
                    grid_json[x][y] = [a.state() for a in grid[x][y]]
                elif (x, y) in debug_cow_path:
                    grid_json[x][y] = [{"type": "*"}]
                else:
                    grid_json[x][y] = []

        return {
            "barn": grid_json,
            "walking_path_heatmap": state["barn"]["walking_path_heatmap"],
            "walking_path_heatmap_max": state["barn"]["walking_path_heatmap_max"],
            "simulation": state["model"],
        }

    def run(self):
        for s in range(self.config["steps"]):
            self.step = s
            for cow in self.cows:
                if cow.alive:
                    cow.step()

            stats_modulo = 100
            if slowmo:
                stats_modulo = 1
                time.sleep(0.05)

            if s % stats_modulo == 0 and self.config["generate_stats"]:
                file = open("stats.txt", "w")
                file.write(self.human_readable_state(self.state()))
                file.close()
                r = json.dumps(self.json_state(self.state()))
                file = open("stats.json", "w")
                file.write(r)
                file.close()


class Barn:
    grid = None
    grass_positions = []
    water_positions = []
    concentrate_positions = []
    sleep_positions = []
    walking_path_heatmap = []

    def __init__(self, config):
        w, h = config["barn_height"], config["barn_width"]
        self.grid = [[[] for x in range(h)] for y in range(w)]
        self.walking_path_heatmap = [[0 for x in range(h)] for y in range(w)]

    def neighborhood(self, pos):
        x, y = pos
        neighborhood = []
        for x_delta, y_delta in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if self.valid_cell((x + x_delta, y + y_delta)):
                neighborhood.append((x + x_delta, y + y_delta))
        return neighborhood

    def neighbors(self, pos, include_center=False):
        x, y = pos
        neighbors = [self.grid[x][y]] if include_center else []
        for pos in self.neighborhood(pos):
            x, y = pos
            neighbors += self.grid[x][y]
        return neighbors

    def valid_cell(self, pos):
        x, y = pos
        if x < 0 or x >= len(self.grid):
            return False
        if y < 0 or y >= len(self.grid[0]):
            return False
        return True

    def is_cell_walkable(self, pos):
        if not self.valid_cell:
            return False
        x, y = pos
        for obj in self.grid[x][y]:
            if isinstance(obj, Cow):
                return False
            if isinstance(obj, Wall):
                return False
        return True

    def is_cell_empty(self, pos):
        x, y = pos
        return len(self.grid[x][y]) == 0

    def move_agent(self, agent, new_pos):
        if tests:
            x, y = new_pos
            assert self.is_cell_walkable(new_pos)
        old_x, old_y = agent.pos
        new_x, new_y = new_pos
        self.grid[old_x][old_y].remove(agent)
        self.grid[new_x][new_y].append(agent)
        agent.update_pos(new_pos)
        if heatmap:
            self.walking_path_heatmap[new_x][new_y] += 1

    def place_agent(self, agent, pos):
        x, y = pos
        # TODO: check
        agent_type = type(agent)
        for agent_x in self.grid[x][y]:
            if type(agent_x) == agent_type:
                print("bug")
                return
        self.grid[x][y].append(agent)
        agent.update_pos(pos)

    def state(self):
        walking_path_heatmap_max = max(
            max(v) for v in [i for i in self.walking_path_heatmap]
        )
        return {
            "grid": self.grid,
            "walking_path_heatmap": self.walking_path_heatmap,
            "walking_path_heatmap_max": walking_path_heatmap_max,
        }


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

    def update_pos(self, pos):
        self.pos = pos

    def update_debug(self, debug):
        self.debug = debug

    def state(self):
        return {"type": self.ASCIIDraw(), "pos": str(self.pos), "weight": self.weight}


class WalkingAgent(Agent):
    alive = True
    moving = False
    stuck_recalc = 0

    current_target = None
    current_path = None

    def __init__(self, model):
        # Ensure Walking Agents always stay on top of other objects
        super().__init__(model, weight=101)

    def random_move(self):
        next_moves = [
            pos
            for pos in self.model.barn.neighborhood(self.pos)
            if self.model.barn.is_cell_walkable(pos)
        ]
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

        if len(self.current_path) == 0 and self.pos != self.current_target:
            # BUG
            self.stuck_recalc += 1
            self.current_objective = None
            # print("bfs thinks you should stay put {}".format(self.pos))
            # print(self.state())
            # self.current_path = None
            return

        if len(self.current_path) == 1:
            # good? [0] is where we are at, sit still
            # self.current_path = None
            return

        self.current_path = self.current_path[1:]
        next_move = self.current_path[0]
        if self.model.barn.is_cell_walkable(next_move):
            self.model.barn.move_agent(self, next_move)
        else:
            self.stuck_recalc += 1
            self.current_objective = None
            # self.random_move()

    def bfs(self, grid, start, target):
        queue = collections.deque([[start]])
        seen = set([start])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x, y) == target:
                return path
            explore = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            random.shuffle(explore)
            for x2, y2 in explore:
                if (x2, y2) in seen:
                    continue
                # ignore cells out of bound
                if not self.model.barn.valid_cell((x2, y2)):
                    continue
                agents = grid[x2][y2]
                if any(type(agent) is OneWayGate for agent in agents):
                    assert len(agents) == 1
                    on_way_gate = list(filter(lambda x: type(x) is OneWayGate, agents))[
                        0
                    ]
                    if not on_way_gate.valid_entry_grid((x, y)):
                        continue
                # we dont want to at onewaygates to "seen", as they can work
                # from another directions
                seen.add((x2, y2))
                # Dont search though walls
                if any(type(agent) is Wall for agent in agents):
                    continue
                # and dont goto cells we have seen before
                # if we search through a onewaygate, check that you are on valid side
                # looks good, lets search further
                queue.append(path + [(x2, y2)])
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
        self.water = random.randrange(
            model.config["max_water"] / 2, model.config["max_water"]
        )
        self.grass = random.randrange(
            model.config["max_grass"] / 2, model.config["max_grass"]
        )
        self.concentrates = random.randrange(
            model.config["max_concentrate"] / 2, model.config["max_concentrate"]
        )
        self.milk = random.randrange(
            model.config["max_milk"] / 2, model.config["max_milk"]
        )

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
        self.concentrates -= use_consentrate_per_step
        # neighbors = self.model.barn.neighbors(self.pos, include_center=True)
        # Test with cow having to be in the grid where the resource is
        x, y = self.pos
        neighbors = self.model.barn.grid[x][y]

        if (
            any(type(agent) is Water for agent in neighbors)
            and self.water < self.model.config["max_water"]
        ):
            self.water += drinks_per_step
        if (
            any(type(agent) is Grass for agent in neighbors)
            and self.grass < self.model.config["max_grass"]
        ):
            self.grass += eats_grass_per_step
        if (
            any(type(agent) is Feeder for agent in neighbors)
            and self.concentrates < self.model.config["max_concentrate"]
        ):
            self.concentrates += eats_consentrate_per_step

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
        if not self.model.barn.is_cell_walkable(self.current_target):
            # if there is another cow on target, choose another one recursivly
            self._update_target(new_objective)
        else:
            self.current_path = self.bfs(
                self.model.barn.grid, self.pos, self.current_target
            )

    def _calc_objective(self):
        # if on the move, easier to change cow's mind
        if self.moving:
            if self.water < 50:
                return "drink"
            if self.concentrates < 10:
                return "eat_concentrates"
            if self.grass < 10:
                return "eat_grass"
        # doing somthing, chaning my mind takes more effort
        else:
            if self.water < 25:
                return "drink"
            if self.concentrates < 5:
                return "eat_concentrates"
            if self.grass < 5:
                return "eat_grass"
        return "sleep"

    def healt(self):
        pass
        # if self.water <= 0 or selg.grass <= 0:

    def state(self):
        return {
            "pos": self.pos,
            "debug": self.debug,
            "alive": self.alive,
            "current_target": self.current_target,
            "current_objective": self.current_objective,
            "water": self.water,
            "concentrates": self.concentrates,
            "grass": self.grass,
            "stuck_recalc": self.stuck_recalc,
            "path": self.current_path,
            "type": self.ASCIIDraw(),
            "weight": self.weight,
        }

    def ASCIIDraw(self):
        if not self.alive:
            return "X"
        return "K"


class Wall(Agent):
    def __init__(self, model):
        super().__init__(model)

    def ASCIIDraw(self):
        return "#"


class Bed(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def ASCIIDraw(self):
        return "B"


class Feeder(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def ASCIIDraw(self):
        return "F"


class Water(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def ASCIIDraw(self):
        return "W"


class Grass(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)

    def ASCIIDraw(self):
        return "G"


class OneWayGate(Agent):
    def __init__(self, model, direction):
        super().__init__(model, weight=99)
        self.direction = direction

    def valid_entry_grid(self, pos):
        gate_pos_x, gate_pos_y = self.pos
        entry_pos_x, entry_pos_y = pos
        if self.direction == "6":
            if gate_pos_x == entry_pos_x - 1 and gate_pos_y == entry_pos_y:
                return True
        if self.direction == "4":
            if gate_pos_x == entry_pos_x + 1 and gate_pos_y == entry_pos_y:
                return True
        if self.direction == "8":
            if gate_pos_x == entry_pos_x and gate_pos_y == entry_pos_y - 1:
                return True
        if self.direction == "2":
            if gate_pos_x == entry_pos_x and gate_pos_y == entry_pos_y + 1:
                return True
        return False

    def ASCIIDraw(self):
        return directions[self.direction]


class SmartGate(Agent):
    def __init__(self, model):
        super().__init__(model, weight=99)
        self.direction = direction

    def ASCIIDraw(self):
        return "O"


def sim(config):
    sim = Simulation(config)
    if profile:
        # todo add worker number awareness and add to file name
        import cProfile

        cProfile.runctx("sim.run()", globals(), locals(), "profile.prof")
    else:
        sim.run()

    rand = bool(random.getrandbits(1))
    return rand


from http.server import HTTPServer, BaseHTTPRequestHandler


class web_server(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        if "png" in self.path:
            try:
                # Reading the file
                file_to_open = open(self.path[1:], "rb").read()
                self.send_response(200)
            except:
                file_to_open = "File not found"
                self.send_response(404)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(bytes(file_to_open))
        else:
            try:
                # Reading the file
                file_to_open = open(self.path[1:]).read()
                self.send_response(200)
            except:
                file_to_open = "File not found"
                self.send_response(404)
            if "json" in self.path:
                self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes(file_to_open, "utf-8"))


def http_server():
    httpd = HTTPServer(("193.35.52.75", 8585), web_server)
    httpd.serve_forever()


if __name__ == "__main__":
    print("There are ", threads, " threads")

    start = time.time()

    jobs = []
    for k in range(threads):
        jobs.append(config.copy())

    jobs[0]["generate_stats"] = True

    http_server = multiprocessing.Process(name="http_server", target=http_server)
    pool = multiprocessing.Pool(threads)

    http_server.start()
    results = pool.map(sim, jobs)
    print(results)

    end = time.time()

    print("The average number of steps is ", int(sum(results) / len(results)), " steps")
    print("The simulation(s) took ", end - start, " seconds to run")

print(
    "Mean:",
    numpy.mean(results),
    " Median:",
    numpy.median(results),
    " Stdev:",
    numpy.std(results),
)
