#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <set>
#include <fann.h>
#include "easygame.h"

using std::min;
using std::max;

const int WIDTH = 1280;
const int HEIGHT = 720;
const int GRID_WIDTH = WIDTH / 20;
const int GRID_HEIGHT = HEIGHT / 20;
const float GRID_CELL_WIDTH = (float)WIDTH / (float)GRID_WIDTH;
const float GRID_CELL_HEIGHT = (float)HEIGHT / (float)GRID_HEIGHT;
const float DT = 1.0f/60.0f;
const float BUDDY_SIZE = 15.0f;
const float FOOD_SIZE = 8.0f;
const int FOOD_COUNT = 50;
const float FOOD_VALUE = 25.0f;
const float EAT_DISTANCE = 20.0f;
const float HEALTH_DECAY = 10.0f;
const float MAX_HEALTH = 100.0f;
const float PLAYER_MOVE_SPEED = 100.0f;

const int NUM_AGENTS = 10;

const int ANN_NUM_LAYERS = 3;
const int ANN_NUM_INPUT = 2;
const int ANN_NUM_HIDDEN = 4;
const int ANN_NUM_OUTPUT = 4;

const int NUM_COLORS = 4;
const float AGENT_COLORS[NUM_COLORS][4] = {
  { 0.3f, 0.5f, 0.2f, 1.0f },
  { 0.3f, 0.2f, 0.5f, 1.0f },
  { 0.5f, 0.3f, 0.2f, 1.0f },
  { 0.5f, 0.3f, 0.5f, 1.0f }
};

enum GridCell {
  GridCellEmpty,
  GridCellFull
};

struct Grid {
  GridCell cells[GRID_WIDTH * GRID_HEIGHT];

  GridCell &cell(int x, int y) {
    assert(x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT);
    return cells[y*GRID_WIDTH + x];
  }

  GridCell &cell_at(float x, float y) {
    assert(x >= 0.0f && x <= WIDTH && y >= 0.0f && y <= HEIGHT);
    int ix = clamp((int)(x / GRID_CELL_WIDTH), 0, GRID_WIDTH - 1);
    int iy = clamp((int)(y / GRID_CELL_HEIGHT), 0, GRID_HEIGHT - 1);
    return cell(ix, iy);
  }
};

struct Food {
  float x, y;
};

Food make_food();

struct AgentInput {
  float nearest_food_dx;
  float nearest_food_dy;
};

struct AgentBehavior {
  float dx, dy;
  bool place_block;
  bool pickup_block;
};

struct Agent {
  float x, y;
  float health;
  fann *ann;
};

void calculate_ann_input(AgentInput input, fann_type ann_input[2]) {
  float mag = sqrtf(input.nearest_food_dx * input.nearest_food_dx +
                    input.nearest_food_dy * input.nearest_food_dy);
  ann_input[0] = input.nearest_food_dx / mag;
  ann_input[1] = input.nearest_food_dy / mag;
}

Agent make_agent() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> fdis(0, 1);

  Agent agent;
  agent.x = WIDTH * fdis(gen);
  agent.y = HEIGHT * fdis(gen);
  agent.health = MAX_HEALTH;
  agent.ann = fann_create_standard(ANN_NUM_LAYERS,
                                       ANN_NUM_INPUT,
                                       ANN_NUM_HIDDEN,
                                       ANN_NUM_OUTPUT);

  fann_set_activation_function_hidden(agent.ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(agent.ann, FANN_SIGMOID_SYMMETRIC);
  fann_randomize_weights(agent.ann, -1.0f, 1.0f);

  return agent;
}

void train_agent(Agent *agent, AgentInput input, AgentBehavior behavior) {
  fann_type ann_input[ANN_NUM_INPUT];
  calculate_ann_input(input, ann_input);

  fann_type ann_output[ANN_NUM_OUTPUT] = {
    behavior.dx / PLAYER_MOVE_SPEED,
    behavior.dy / PLAYER_MOVE_SPEED,
    behavior.place_block ? 1.0f : 0.0f,
    behavior.pickup_block ? 1.0f : 0.0f
  };

  fann_train(agent->ann, ann_input, ann_output);
}

AgentBehavior run_agent(Agent *agent, AgentInput input) {
  fann_type ann_input[2];
  calculate_ann_input(input, ann_input);

  fann_type *ann_output = fann_run(agent->ann, ann_input);

  AgentBehavior b = {
    PLAYER_MOVE_SPEED * ann_output[0],
    PLAYER_MOVE_SPEED * ann_output[1],
    ann_output[2] > 0.5f, // place block
    ann_output[3] > 0.5f  // pickup block
  };
  return b;
}

void print_ann(fann *ann) {
  int num_conn = fann_get_total_connections(ann);
  fann_connection connections[num_conn];
  fann_get_connection_array(ann, connections);
  for(int i = 0; i < num_conn; i++) {
    printf("%d -> %d: %f\n",
           connections[i].from_neuron,
           connections[i].to_neuron,
           connections[i].weight);
  }
}

int main(int argc, char *argv[]) {
  assert(NUM_AGENTS >= 1);

  eg_init(WIDTH, HEIGHT, "Buddies");

  Grid grid;
  Agent agents[NUM_AGENTS];
  Food foods[FOOD_COUNT];

  for(int y = 0; y < GRID_HEIGHT; y++) {
    for(int x = 0; x < GRID_WIDTH; x++) {
      grid.cell(x, y) = GridCellEmpty;
    }
  }

  for(int i = 0; i < NUM_AGENTS; i++) {
    agents[i] = make_agent();
  }

  for(int i = 0; i < FOOD_COUNT; i++) {
    foods[i] = make_food();
  }

  bool quit = false;
  while(!quit) {
    EGEvent event;
    while(eg_poll_event(&event)) {
      if(event.type == SDL_QUIT) {
        quit = true;
      }
    }

    AgentInput agent_inputs[NUM_AGENTS];
    for(int i = 0; i < NUM_AGENTS; i++) {
      int nearest_index;
      float nearest_dist_sq;
      for(int j = 0; j < FOOD_COUNT; j++) {
        float dx = foods[j].x - agents[i].x;
        float dy = foods[j].y - agents[i].y;
        float dist_sq = (dx*dx) + (dy*dy);
        if(j == 0 || dist_sq < nearest_dist_sq) {
          nearest_index = j;
          nearest_dist_sq = dist_sq;
        }
      }
      agent_inputs[i].nearest_food_dx = foods[nearest_index].x - agents[i].x;
      agent_inputs[i].nearest_food_dy = foods[nearest_index].y - agents[i].y;
    }

    AgentBehavior agent_behaviors[NUM_AGENTS];

    // player's agent behavior
    agent_behaviors[0].dx = 0.0f;
    agent_behaviors[0].dy = 0.0f;
    if(eg_get_keystate(SDL_SCANCODE_LEFT))  agent_behaviors[0].dx -= PLAYER_MOVE_SPEED;
    if(eg_get_keystate(SDL_SCANCODE_RIGHT)) agent_behaviors[0].dx += PLAYER_MOVE_SPEED;
    if(eg_get_keystate(SDL_SCANCODE_DOWN))  agent_behaviors[0].dy -= PLAYER_MOVE_SPEED;
    if(eg_get_keystate(SDL_SCANCODE_UP))    agent_behaviors[0].dy += PLAYER_MOVE_SPEED;
    agent_behaviors[0].place_block = eg_get_keystate(SDL_SCANCODE_SPACE);
    agent_behaviors[0].pickup_block = false;

    // other agents' behaviors and training
    for(int i = 1; i < NUM_AGENTS; i++) {
      // if the player's agent is moving, we'll train everyone to mimic the player
      if(!(agent_behaviors[0].dx == 0.0f && agent_behaviors[0].dy == 0.0f)) {
        train_agent(&agents[i], agent_inputs[0], agent_behaviors[0]);
      }
      agent_behaviors[i] = run_agent(&agents[i], agent_inputs[i]);
    }

    for(int i = 0; i < NUM_AGENTS; i++) {
      agents[i].x = clamp(agents[i].x + DT * agent_behaviors[i].dx, 0.0f, (float)WIDTH);
      agents[i].y = clamp(agents[i].y + DT * agent_behaviors[i].dy, 0.0f, (float)HEIGHT);
      agents[i].health = max(0.0f, agents[i].health - DT * HEALTH_DECAY);

      for(int j = 0; j < FOOD_COUNT; j++) {
         float dx = foods[j].x - agents[i].x;
         float dy = foods[j].y - agents[i].y;
         float dist_sq = (dx*dx) + (dy*dy);
         if(dist_sq < EAT_DISTANCE*EAT_DISTANCE) {
           agents[i].health = min(MAX_HEALTH, agents[i].health + FOOD_VALUE);
           foods[j] = make_food();
           break;
         }
      }

      if(agent_behaviors[i].place_block) {
        if(grid.cell_at(agents[i].x, agents[i].y) == GridCellEmpty) {
          grid.cell_at(agents[i].x, agents[i].y) = GridCellFull;
        }
      }

      if(agents[i].health <= 0.0f) {
        if(agents[i].ann) fann_destroy(agents[i].ann);
        agents[i] = make_agent();
      }
    }

    eg_clear_screen(0.7f, 0.8f, 0.75f, 0.0f);

    // draw grid
    eg_set_color(0.6f, 0.7f, 0.65f, 1.0f);
    for(int x = 0; x < GRID_WIDTH; x++) {
      float fx = x / (float)GRID_WIDTH * WIDTH;
      eg_draw_line(fx, 0.0f, fx, HEIGHT);
    }
    for(int y = 0; y < GRID_HEIGHT; y++) {
      float fy = y / (float)GRID_WIDTH * WIDTH;
      eg_draw_line(0.0f, fy, WIDTH, fy);
    }
    for(int y = 0; y < GRID_HEIGHT; y++) {
      for(int x = 0; x < GRID_WIDTH; x++) {
        if(grid.cell(x, y) == GridCellFull) {
          float fx = x / (float)GRID_WIDTH * WIDTH;
          float fy = y / (float)GRID_WIDTH * WIDTH;
          float fw = WIDTH / (float)GRID_WIDTH;
          float fh = HEIGHT / (float)GRID_HEIGHT;
          eg_draw_square(fx, fy, fw, fh);
        }
      }
    }

    // draw agents
    for(int i = 0; i < NUM_AGENTS; i++) {
      eg_set_color(AGENT_COLORS[i % NUM_COLORS][0],
                   AGENT_COLORS[i % NUM_COLORS][1],
                   AGENT_COLORS[i % NUM_COLORS][2],
                   AGENT_COLORS[i % NUM_COLORS][3]);    
      eg_draw_square(agents[i].x - 0.5f*BUDDY_SIZE, agents[i].y - 0.5f*BUDDY_SIZE, BUDDY_SIZE, BUDDY_SIZE);

      eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
      eg_draw_square(agents[i].x - 15.0f, agents[i].y + 12.0f, 30.0f, 5.0f);
      eg_set_color(1.0f, 1.0f, 0.5f, 1.0f);
      eg_draw_square(agents[i].x - 15.0f, agents[i].y + 12.0f, agents[i].health * 30.0f / MAX_HEALTH, 5.0f);

      eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
      eg_draw_line(agents[i].x,
                   agents[i].y,
                   agents[i].x + agent_inputs[i].nearest_food_dx,
                   agents[i].y + agent_inputs[i].nearest_food_dy);
    }

    // draw foods
    eg_set_color(0.8f, 0.5f, 0.1f, 1.0f);
    for(int i = 0; i < FOOD_COUNT; i++) {
      eg_draw_square(foods[i].x - 0.5f*FOOD_SIZE, foods[i].y - 0.5f*FOOD_SIZE, FOOD_SIZE, FOOD_SIZE);
    }

    eg_swap_buffers();
  }

  eg_shutdown();

  return 0;
}

Food make_food() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> fdis(0, 1);
  Food f = { WIDTH * fdis(gen), HEIGHT * fdis(gen) };
  return f;
}
