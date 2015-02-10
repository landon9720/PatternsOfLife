#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <set>
#include <fann.h>
#ifdef __EMSCRIPTEN__
#include "easygame_emscripten.h"
#include <emscripten.h>
#else
#include "easygame.h"
#endif


using std::min;
using std::max;

const int   TURBO_RATE = 240; // how many simulation steps per render
const int   WIDTH = 1280;
const int   HEIGHT = 720;
const int   GRID_WIDTH = WIDTH / 20;
const int   GRID_HEIGHT = HEIGHT / 20;
const float GRID_CELL_WIDTH = (float)WIDTH / (float)GRID_WIDTH;
const float GRID_CELL_HEIGHT = (float)HEIGHT / (float)GRID_HEIGHT;
const float DT = 1.0f/60.0f;
const float BUDDY_SIZE = 10.0f;
const float FOOD_SIZE = 6.0f;
const int   FOOD_COUNT = 400;
const float FOOD_VALUE = 100.0f;
const float FLOW_DX = -0.100f;
const float FLOW_DY = +0.005f;
const float EAT_DISTANCE = 20.0f;
const float HEALTH_DECAY = 0.2f;
const float HEALTH_DECAY_CONSTANT = 0.01f;
const float MAX_HEALTH = 100.0f;
const float AGENT_MAX_FORCE = 100.0f;
const float AGENT_MAX_ROTATIONAL_FORCE = M_PI / 40.0f;
const float LEARNING_RATE = 0.010f;

const int NUM_AGENTS = 10;

// in  1: radians to food
// in  2: distance to food
// in  3: health of self
// out 1: move dx
// out 2: move dy

const int ANN_NUM_INPUT = 4;
const int ANN_NUM_HIDDEN = 7;
const int ANN_NUM_OUTPUT = 2;

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
  float dx, dy;
  float value;
};

Food make_food();

struct AgentInput {
  float nearest_food_relative_direction;
  float nearest_food_distance;
  float self_health;
};

struct AgentBehavior {
  float rotational_force, force;
};

struct Agent {
  float x, y;
  float orientation;
  float health;
  int score;
  fann *ann;
};

void calculate_ann_input(AgentInput input, fann_type ann_input[ANN_NUM_INPUT]) {
  ann_input[0] = input.nearest_food_relative_direction;
  ann_input[1] = input.nearest_food_distance;
  ann_input[2] = input.self_health;
}

Agent make_agent() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> fdis(0, 1);

  Agent agent;
  // 1. agents start in a fixed location
  agent.x = WIDTH * 0.25f;
  agent.y = HEIGHT * 0.25f;
  // 2. -or- agents start in a random location
//  agent.x = WIDTH * fdis(gen);
//  agent.y = HEIGHT * fdis(gen);
  agent.orientation = 0.0f;
  agent.orientation = fdis(gen) * 2 * M_PI - M_PI;
  agent.health = MAX_HEALTH;
  agent.score = 0;
  agent.ann = fann_create_standard(4, ANN_NUM_INPUT, ANN_NUM_HIDDEN, ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);
  fann_set_activation_function_hidden(agent.ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(agent.ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_training_algorithm(agent.ann, FANN_TRAIN_INCREMENTAL);
  fann_set_learning_rate(agent.ann, LEARNING_RATE);
  fann_randomize_weights(agent.ann, -1.0f, 1.0f);
  return agent;
}

void train_agent(Agent *agent, AgentInput input, AgentBehavior behavior) {
  fann_type ann_input[ANN_NUM_INPUT];
  calculate_ann_input(input, ann_input);

  fann_type ann_output[ANN_NUM_OUTPUT] = {
    behavior.rotational_force / AGENT_MAX_ROTATIONAL_FORCE,
    behavior.force / AGENT_MAX_FORCE
  };

  fann_train(agent->ann, ann_input, ann_output);
}

AgentBehavior run_agent(Agent *agent, AgentInput input) {
  fann_type ann_input[ANN_NUM_INPUT];
  calculate_ann_input(input, ann_input);

  fann_type *ann_output = fann_run(agent->ann, ann_input);

  AgentBehavior b = {
    AGENT_MAX_ROTATIONAL_FORCE * ann_output[0],
    AGENT_MAX_FORCE * ann_output[1]
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

static int frame = 0;
static Grid grid;
static Agent agents[NUM_AGENTS];
static Food foods[FOOD_COUNT];
static bool quit = false;
static EGSound *pickup_sound;

void init() {
  eg_init(WIDTH, HEIGHT, "Buddies");

  pickup_sound = eg_load_sound("assets/pickup.wav");

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
}

void step() {
  EGEvent event;
  while(eg_poll_event(&event)) {
    if(event.type == SDL_QUIT) {
      quit = true;
    }
  }

  // food model
  for(int i = 0; i < FOOD_COUNT; i++) {
    foods[i].x += foods[i].dx;
    foods[i].y += foods[i].dy;
    if (foods[i].x < 0.0f || foods[i].x > WIDTH || foods[i].y < 0.0f || foods[i].y > HEIGHT) {
      foods[i] = make_food();
    }
  }

  // agent model
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
    float dx = foods[nearest_index].x - agents[i].x;
    float dy = foods[nearest_index].y - agents[i].y;
    agent_inputs[i].nearest_food_relative_direction = atan2(dy, dx) - agents[i].orientation;
    agent_inputs[i].nearest_food_distance = sqrtf(dx * dx + dy * dy);
    agent_inputs[i].self_health = agents[i].health;
  }

  // index of high scoring agent
  int high_score_index = 0;
  for(int i = 1; i < NUM_AGENTS; i++) {
    if(agents[i].score > agents[high_score_index].score) {
      high_score_index = i;
    }
  }

  AgentBehavior agent_behaviors[NUM_AGENTS];

  for(int i = 0; i < NUM_AGENTS; i++) {

    //
    //
    if (i != high_score_index) { // don't train the leader
      train_agent(&agents[i], agent_inputs[high_score_index], agent_behaviors[high_score_index]);
    }

    //
    //
    agent_behaviors[i] = run_agent(&agents[i], agent_inputs[i]);

    agents[i].orientation = agents[i].orientation + agent_behaviors[i].rotational_force;
    agents[i].x = agents[i].x + DT * agent_behaviors[i].force * (float)cos(agents[i].orientation);
    agents[i].y = agents[i].y + DT * agent_behaviors[i].force * (float)sin(agents[i].orientation);

    // decay health as a function of force and time
    float new_health = agents[i].health - ((DT * HEALTH_DECAY * abs(agent_behaviors[i].force)) + HEALTH_DECAY_CONSTANT);
    agents[i].health = max(0.0f, new_health);

    for(int j = 0; j < FOOD_COUNT; j++) {
       float dx = foods[j].x - agents[i].x;
       float dy = foods[j].y - agents[i].y;
       float dist_sq = (dx*dx) + (dy*dy);
       if(dist_sq < EAT_DISTANCE*EAT_DISTANCE) {
         agents[i].health = min(MAX_HEALTH, agents[i].health + FOOD_VALUE * foods[j].value);
         agents[i].score++;
         foods[j] = make_food();
         eg_play_sound(pickup_sound);
         break;
       }
    }

    // death
    if(agents[i].health <= 0.0f) {
      if(agents[i].ann) fann_destroy(agents[i].ann);
      agents[i] = make_agent();
    }
  }

  bool skip_render = eg_get_keystate(SDL_SCANCODE_F) && (frame % TURBO_RATE != 0);
  if(!skip_render) {
    eg_clear_screen(0.0f, 0.0f, 0.0f, 0.0f);

    // draw grid
    eg_set_color(0.5f, 0.5f, 0.5f, 0.5f);
    for(int x = 0; x < GRID_WIDTH; x++) {
      float fx = x / (float)GRID_WIDTH * WIDTH;
      eg_draw_line(fx, 0.0f, fx, HEIGHT);
      if (x % 7 == 0) {
        eg_set_color(1.0f, 1.0f, 1.0f, 0.3f);
        eg_draw_line(fx, 0.0f, fx, HEIGHT);
        eg_set_color(0.5f, 0.5f, 0.5f, 0.5f);
      }
    }
    for(int y = 0; y < GRID_HEIGHT; y++) {
      float fy = y / (float)GRID_WIDTH * WIDTH;
      eg_draw_line(0.0f, fy, WIDTH, fy);
      if (y % 7 == 0) {
        eg_set_color(1.0f, 1.0f, 1.0f, 0.3f);
        eg_set_color(0.5f, 0.5f, 0.5f, 0.5f);
      }
    }

    // draw agents
    for(int i = 0; i < NUM_AGENTS; i++) {

      // indicate orientation
      eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
      eg_draw_line(agents[i].x,
                   agents[i].y,
                   agents[i].x + (float)cos(agents[i].orientation) * 10.0F,
                   agents[i].y + (float)sin(agents[i].orientation) * 10.0F,
                   10);

      // agent
      eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
      eg_draw_square(agents[i].x - 0.5f*BUDDY_SIZE, agents[i].y - 0.5f*BUDDY_SIZE, BUDDY_SIZE, BUDDY_SIZE);
      eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
      eg_draw_square(agents[i].x - 0.5f*BUDDY_SIZE*0.5f, agents[i].y - 0.5f*BUDDY_SIZE*0.5f, BUDDY_SIZE*0.5f, BUDDY_SIZE*0.5f);

      // health bar
      eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
      eg_draw_square(agents[i].x - 15.0f, agents[i].y + 12.0f, 30.0f, 5.0f);
      if (agents[i].health > MAX_HEALTH * 0.25f) {
        eg_set_color(0.5f, 0.9f, 0.5f, 0.8f);
      } else {
        eg_set_color(0.8f, 0.3f, 0.3f, 0.8f);
      }
      eg_draw_square(agents[i].x - 15.0f, agents[i].y + 12.0f, agents[i].health * 30.0f / MAX_HEALTH, 5.0f);

      // score bar
      eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
      eg_draw_square(agents[i].x - 15.0f, agents[i].y + 20.0f, 30.0f, 5.0f);
      eg_set_color(0.9f, 0.85f, 0.0f, 0.8f);
      eg_draw_square(agents[i].x - 15.0f, agents[i].y + 20.0f, 30.0f * ((float)agents[i].score / (float)agents[high_score_index].score), 5.0f);
    }

    // draw foods
    for(int i = 0; i < FOOD_COUNT; i++) {
      eg_set_color(0.0f, 0.8f, 0.0f, 1.0f - 0.5f * foods[i].value);
      eg_draw_square(foods[i].x - 0.5f*FOOD_SIZE, foods[i].y - 0.5f*FOOD_SIZE, FOOD_SIZE, FOOD_SIZE);
    }

    // high score
    eg_set_color(0.9f, 0.3f, 0.3f, 1.0f);
    eg_draw_square(agents[high_score_index].x - 0.5f*BUDDY_SIZE, agents[high_score_index].y - 0.5f*BUDDY_SIZE, BUDDY_SIZE, BUDDY_SIZE);

    eg_swap_buffers();
  }

  frame++;
}

int main(int argc, char *argv[]) {
  assert(NUM_AGENTS >= 1);

  init();

#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop(step, 0, 1);
#else
  while(!quit) {
    step();
  }
#endif

  eg_shutdown();

  return 0;
}

Food make_food() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> fdis(0, 1);
  float x = WIDTH * fdis(gen);
  float y = HEIGHT * fdis(gen);
  float dx = FLOW_DX * 0.5f + x / WIDTH * FLOW_DX * 0.5f;
  float dy = FLOW_DY * 0.5f + x / HEIGHT * FLOW_DY;
  float value = x / WIDTH;
  Food f = {
    x, y,
    dx, dy,
    value
  };
  return f;
}
