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
const int   WIDTH = 1280;///2;
const int   HEIGHT = 720;///2;
const int   GRID_WIDTH = WIDTH / 20;
const int   GRID_HEIGHT = HEIGHT / 20;
const float GRID_CELL_WIDTH = (float)WIDTH / (float)GRID_WIDTH;
const float GRID_CELL_HEIGHT = (float)HEIGHT / (float)GRID_HEIGHT;
const float BUDDY_SIZE = 10.0f;
const float FOOD_SIZE = 6.0f;
const int   FOOD_COUNT = 100;
const float FOOD_VALUE = 100.0f;
const float FLOW_DX = -0.002f;
const float FLOW_DY = +0.001f;
const float EAT_DISTANCE = 20.0f;
const float HEALTH_DECAY = 0.2f;
const float HEALTH_DECAY_CONSTANT = 0.2f;
const float MAX_HEALTH = 100.0f;
const float LEARNING_RATE = 0.01f;
const int   RECORD_SAMPLE_RATE = 1;

const int   NUM_AGENTS = 1;

const int   ANN_NUM_INPUT = 2;
const int   ANN_NUM_HIDDEN = 10;
const int   ANN_NUM_OUTPUT = 2;
const int   ANN_NUM_CONNECTIONS = 52; // how to calculate this?

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> fdis(0, 1);

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

struct AgentInput {
  float nearest_food_relative_direction;
  float nearest_food_distance;
};

struct AgentBehavior {
  float rotational ; // rotational (twising and turning) behavior of the agent
  float linear     ; // forward moving (in the direction of orientation) behavior of the agent
};

struct Agent {
  int parent_index;
  float x, y;
  float orientation;
  float health;
  int score;
  fann *ann;
};

struct Record {
  float weights[ANN_NUM_CONNECTIONS];
  int total_score;
};

void init_agent(Agent *agent) {
  agent->parent_index = -1;
  agent->x = WIDTH * fdis(gen);
  agent->y = HEIGHT * fdis(gen);
  agent->orientation = fdis(gen) * 2 * M_PI - M_PI;
  agent->health = MAX_HEALTH;
  agent->score = 0;
}

Food make_food() {
  float x = WIDTH * fdis(gen);
  float y = HEIGHT * fdis(gen);
  float dx = FLOW_DX * 0.5f + x / WIDTH * FLOW_DX * 0.5f;
  float dy = FLOW_DY * 0.5f + x / HEIGHT * FLOW_DY;
  Food f = {
    x, y,
    dx, dy
  };
  return f;
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
  printf("MSE: %f\n", fann_get_MSE(ann));
}

static        int frame               = 0     ;
static       Grid grid                        ;
static      Agent agents[NUM_AGENTS]          ;
static       Food foods[FOOD_COUNT]           ;
static       bool quit                = false ;
static    EGSound *pickup_sound               ;
static     Record records[WIDTH]              ;
static        int records_index       = 0     ;

void init() {

  eg_init(WIDTH, HEIGHT, "Buddies");

  pickup_sound = eg_load_sound("assets/pickup.wav");

  for(int y = 0; y < GRID_HEIGHT; y++) {
    for(int x = 0; x < GRID_WIDTH; x++) {
      grid.cell(x, y) = GridCellEmpty;
    }
  }

  for(int i = 0; i < NUM_AGENTS; i++) {
    init_agent(&agents[i]);
    agents[i].ann = fann_create_standard(3, ANN_NUM_INPUT, ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);
    fann_set_activation_function_hidden(agents[i].ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(agents[i].ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_training_algorithm(agents[i].ann, FANN_TRAIN_INCREMENTAL);
    fann_set_learning_rate(agents[i].ann, LEARNING_RATE);
    fann_randomize_weights(agents[i].ann, -.10f, +.10f);
  }

  for(int i = 0; i < FOOD_COUNT; i++) {
    foods[i] = make_food();
  }

  printf("number of connections: %d (%d)\n", fann_get_total_connections(agents[0].ann), ANN_NUM_CONNECTIONS);
  assert(fann_get_total_connections(agents[0].ann) == ANN_NUM_CONNECTIONS);

}

void step() {

  // handle user events
  EGEvent event;
  while (eg_poll_event(&event)) {
    if (event.type == SDL_QUIT) {
      quit = true;
    } else if (event.type == SDL_MOUSEBUTTONDOWN) {
      SDL_MouseButtonEvent e = event.button;
      printf("SDL_MOUSEBUTTONDOWN button=%d state=%d x=%d y=%d\n", e.button, e.state, e.x, e.y);
    }
  }

  // update food model
  for(int i = 0; i < FOOD_COUNT; i++) {
    foods[i].x += foods[i].dx;
    foods[i].y += foods[i].dy;
    if (foods[i].x < 0.0f || foods[i].x > WIDTH || foods[i].y < 0.0f || foods[i].y > HEIGHT) {
      foods[i] = make_food();
    }
  }

  // food index maps every agent to its nearest food
  int food_index[NUM_AGENTS];
  for (int i = 0; i < NUM_AGENTS; i++) {
    float nearest_dist_sq;
    for (int j = 0; j < FOOD_COUNT; j++) {
      float dx = foods[j].x - agents[i].x;
      float dy = foods[j].y - agents[i].y;
      float dist_sq = (dx*dx) + (dy*dy);
      if (j == 0 || dist_sq < nearest_dist_sq) {
        food_index[i] = j;
        nearest_dist_sq = dist_sq;
      }
    }
  }

  // map the food index to the agent input model
  AgentInput agent_inputs[NUM_AGENTS];
  for (int i = 0; i < NUM_AGENTS; i++) {
    float dx = foods[food_index[i]].x - agents[i].x;
    float dy = foods[food_index[i]].y - agents[i].y;
    agent_inputs[i].nearest_food_relative_direction = angle_diff(atan2(dy, dx), agents[i].orientation);
    agent_inputs[i].nearest_food_distance = sqrtf(dx * dx + dy * dy);
    printf("agent_inputs[i].nearest_food_relative_direction=%f\n", agent_inputs[i].nearest_food_relative_direction);
    printf("agent_inputs[i].nearest_food_distance=%f\n", agent_inputs[i].nearest_food_distance);
  }

  // index of high scoring agent
  int high_score_index = 0;
  for (int i = 1; i < NUM_AGENTS; i++) {
    if (agents[i].score > agents[high_score_index].score) {
      high_score_index = i;
    }
  }

  printf("ANN\n");
  print_ann(agents[high_score_index].ann);

  // index of selected agent
  int selected_index = 0;
  int total_score = 0;
  for (int i = 0; i < NUM_AGENTS; i++) {
    total_score += agents[i].score;
  }
  int random_score = (int)(fdis(gen) * (float)total_score);
  for (int i = 0; i < NUM_AGENTS && random_score >= 0.0f; i++) {
    random_score -= agents[i].score;
    selected_index = i;
  }

  //// the behavior model
  ///
  //
  //
  //
  //
  for (int i = 0; i < NUM_AGENTS; i++) {

    float ann_input[ANN_NUM_INPUT];
    ann_input[0] = agent_inputs[i].nearest_food_relative_direction;
    ann_input[1] = agent_inputs[i].nearest_food_distance;
    printf("ann_input[0]=%f\nann_input[1]=%f\n", ann_input[0], ann_input[1]);

    float *ann_output = fann_run(agents[i].ann, ann_input);
    printf("ann_output[0]=%f\nann_output[1]=%f\n", ann_output[0], ann_output[1]);

    AgentBehavior b = {
      ann_output[0],
      ann_output[1]
    };
    printf("b.rotational=%f\nb.linear=%f\n", b.rotational, b.linear);

    //
    //
    // train via mouse
    int mouse_x, mouse_y;
    SDL_PumpEvents();
    if (SDL_GetMouseState(&mouse_x, &mouse_y) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      float dx_to_mouse =  (float)mouse_x - agents[i].x;
      float dy_to_mouse = ((float)HEIGHT - (float)mouse_y) - agents[i].y;
      float fog_rotation = atan2(dy_to_mouse, dx_to_mouse);
      float delta_radians = angle_diff(fog_rotation, agents[i].orientation);
      float distance = sqrtf(dx_to_mouse * dx_to_mouse + dy_to_mouse * dy_to_mouse);
      printf("delta_radians=%f\n", delta_radians);
      printf("distance=%f\n", distance);
      float fog_rotational_behavior = max(min(delta_radians, +1.0f), -1.0f);
      float fog_linear_behavior = max(min(distance, +1.0f), -1.0f);
      printf("fog_rotational_behavior=%f\n", fog_rotational_behavior);
      printf("fog_linear_behavior=%f\n", fog_linear_behavior);
      float ann_output_train[ANN_NUM_OUTPUT] = {
        fog_rotational_behavior,
        fog_linear_behavior
      };
      fann_train(agents[i].ann, ann_input, ann_output_train);
      printf("ann_output_train[0]=%f\nann_output_train[1]=%f\n", ann_output_train[0], ann_output_train[1]);
    }

    // update agent orientation and position
    agents[i].orientation = angle_diff(agents[i].orientation + b.rotational, 0.0f);
    agents[i].x = agents[i].x + b.linear * (float)cos(agents[i].orientation);
    agents[i].y = agents[i].y + b.linear * (float)sin(agents[i].orientation);

    // decay health as a function of time
    agents[i].health = max(0.0f, agents[i].health - HEALTH_DECAY_CONSTANT);

    for (int j = 0; j < FOOD_COUNT; j++) {
       float dx = foods[j].x - agents[i].x;
       float dy = foods[j].y - agents[i].y;
       float dist_sq = (dx*dx) + (dy*dy);
       if(dist_sq < EAT_DISTANCE*EAT_DISTANCE) {
         agents[i].health = min(MAX_HEALTH, agents[i].health + FOOD_VALUE * foods[j].value);
         agents[i].score++;
         foods[j] = make_food();
//         eg_play_sound(pickup_sound);
         break;
       }
    }

    // death
    if (agents[i].health <= 0.0f) {

      for (int j = 0; j < NUM_AGENTS; j++) {
        if (agents[j].parent_index == i) {
          agents[j].parent_index = -1;
        }
      }

      init_agent(&agents[i]);

      // if (agents[selected_index].score > 0) {
      //   agents[i].parent_index = selected_index;
      //   agents[i].x = agents[selected_index].x;
      //   agents[i].y = agents[selected_index].y;
      //   fann *source_ann = agents[selected_index].ann;
      //   fann *target_ann = agents[i].ann;
      //   int num_conn = fann_get_total_connections(source_ann);
      //   fann_connection source_connections[num_conn];
      //   fann_get_connection_array(source_ann, source_connections);
      //   fann_connection target_connections[num_conn];
      //   fann_get_connection_array(target_ann, target_connections);
      //   target_connections[i].weight = source_connections[i].weight;
      //   fann_set_weight_array(target_ann, target_connections, num_conn);
      // } else {
        fann_randomize_weights(agents[i].ann, -.10f, +.10f);
      // }
    }
  }

  // update the record model
  if (frame % RECORD_SAMPLE_RATE == 0) {
    records_index++;
    records_index %= WIDTH;
    fann *ann = agents[high_score_index].ann;
    int num_conn = fann_get_total_connections(ann);
    fann_connection connections[num_conn];
    fann_get_connection_array(ann, connections);
    for (int i = 0; i < num_conn; i++) {
      records[records_index].weights[i] = connections[i].weight;
    }
    records[records_index].total_score = total_score;
  }

  // f               for 4% turbo mode
  // lshift-f        for 20% turbo mode
  // lshift-rshift-f for 100% turbo mode
  bool skip_render = false;
  if (eg_get_keystate(SDL_SCANCODE_F)) {
    int rate = int((float)TURBO_RATE * 0.04f);
    bool l_shift = eg_get_keystate(SDL_SCANCODE_LSHIFT);
    bool r_shift = eg_get_keystate(SDL_SCANCODE_RSHIFT);
    if      (l_shift && ! r_shift ) rate = int((float)TURBO_RATE * 0.20f);
    else if (l_shift &&   r_shift)  rate = TURBO_RATE;
    skip_render = (frame % rate != 0);
  }

  if (!skip_render) {
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
        eg_draw_line(0.0f, fy, WIDTH, fy);
        eg_set_color(0.5f, 0.5f, 0.5f, 0.5f);
      }
    }

    ///
    //
    //
    //
    // draw agents
    for (int i = 0; i < NUM_AGENTS; i++) {

      // indicate orientation
      eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
      eg_draw_line(agents[i].x,
                   agents[i].y,
                   agents[i].x + (float)cos(agents[i].orientation) * 10.0F,
                   agents[i].y + (float)sin(agents[i].orientation) * 10.0F,
                   10);


      // indicate nearest food (by food index)
      eg_set_color(1.0f, 1.0f, 1.0f, 0.4f);
      eg_draw_line(agents[i].x,
                   agents[i].y,
                   foods[food_index[i]].x,
                   foods[food_index[i]].y,
                   2);

      // indicate nearest food (by sensor)             
      eg_draw_line(agents[i].x,
                   agents[i].y,
                   agents[i].x + (float)cos(agents[i].orientation + agent_inputs[i].nearest_food_relative_direction) * agent_inputs[i].nearest_food_distance,
                   agents[i].y + (float)sin(agents[i].orientation + agent_inputs[i].nearest_food_relative_direction) * agent_inputs[i].nearest_food_distance,
                   2);

      // agent
      float buddy_size = BUDDY_SIZE * (agents[i].parent_index == -1 ? 1.0f : 0.6f);
      if (i == selected_index) {
        float highlight_size = buddy_size * 1.2f;
        eg_set_color(0.99f, 0.99f, 0.01f, 0.9f);
        eg_draw_square(agents[selected_index].x - 0.5f*highlight_size, agents[selected_index].y - 0.5f*highlight_size, highlight_size, highlight_size);
      }
      eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
      eg_draw_square(agents[i].x - 0.5f*buddy_size, agents[i].y - 0.5f*buddy_size, buddy_size, buddy_size);
      eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
      eg_draw_square(agents[i].x - 0.5f*buddy_size*0.5f, agents[i].y - 0.5f*buddy_size*0.5f, buddy_size*0.5f, buddy_size*0.5f);

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

    // record model
    if (eg_get_keystate(SDL_SCANCODE_TAB)) {
      for (int rx = 0; rx < WIDTH; rx++) {
        float y = 0.0f;
        for (int wi = 0; wi < ANN_NUM_CONNECTIONS; wi++) {
          // weight graph
          if (wi % 2 == 0) eg_set_color(0.8f, 0.8f, 0.8f, 1.0f);
          else             eg_set_color(0.5f, 0.5f, 0.5f, 1.0f);
          float w = records[rx].weights[wi];
          float h;
          if (w > 0.0f) h = w * 30.0f;
          else h = log(-w * 30.0f);
          eg_draw_line(rx, y, rx, y + h, 1.5f);
          y += h;
          // total score graph
          eg_set_color(0.8f, 0.8f, 0.8f, 1.0f);
          eg_draw_line(rx, HEIGHT, rx, HEIGHT - records[rx].total_score, 1.5f);
        }
      }
    }

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
