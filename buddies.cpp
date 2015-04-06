#include "buddies.h"

struct Agent;

struct WorldHex {
  bool food;
  Agent *agent;
};

const int HEX_SIZE = 15;
const int WIDTH = 1280;
const int HEIGHT = 720;
const int Q = 200 * 2;
const int R = 123 * 2;
const int WORLD_SIZE = Q * R;

const int ANN_NUM_INPUT = 5;
const int ANN_NUM_HIDDEN = 7;
const int ANN_NUM_OUTPUT = 4;
const int ANN_NUM_CONNECTIONS = 130; // how to calculate this? ¯\_(ツ)_/¯

const int DNA_SIZE = ANN_NUM_CONNECTIONS + 5;

static bool draw_extra_info = false;
static int following = -1;
static bool moving = false;
static bool nudge = false;
static bool pause = false;
static bool quit = false;
static bool zooming = false;
static float camera_x = HEX_SIZE * R;
static float camera_y = HEX_SIZE * Q;
static float camera_zoom = 0.5f;
static float food_spawn_rate = 0.001f;
static int draw_record = 0;
static int frame = 0;
static int frame_rate = 1;
static int moving_home_x;
static int moving_home_y;
static int num_agents = 20;
static int records_index = 0;
static int zooming_home;

void cubic_to_axial(int x, int y, int z, int &q, int &r) {
  q = x;
  r = z;
}

void axial_to_cubic(int q, int r, int &x, int &y, int &z) {
  x = q;
  z = r;
  y = -x - z;
}

static WorldHex world[WORLD_SIZE];

WorldHex *hex_axial(int q, int r) {
  if (q < 0 || q >= Q || r < 0 || r >= R) {
    return 0;
  }
  return &world[q + r * Q];
}

WorldHex *hex_cubic(int x, int y, int z) {
  int q, r;
  cubic_to_axial(x, y, z, q, r);
  return hex_axial(q, r);
}

void axial_to_xy(int q, int r, int &x, int &y) {
  x = HEX_SIZE * 3.0f / 2.0f * q;
  y = HEX_SIZE * sqrtf(3.0f) * (r + q / 2.0f);
}

void cubic_add(int &x, int &y, int &z, int dx, int dy, int dz) {
  x = x + dx;
  y = y + dy;
  z = z + dz;
}

void cubic_add_direction(int &x, int &y, int &z, int direction) {
  switch (direction) {
  case 0:
    cubic_add(x, y, z, 1, -1, 0);
    break;
  case 1:
    cubic_add(x, y, z, 0, -1, 1);
    break;
  case 2:
    cubic_add(x, y, z, -1, 0, 1);
    break;
  case 3:
    cubic_add(x, y, z, -1, 1, 0);
    break;
  case 4:
    cubic_add(x, y, z, 0, 1, -1);
    break;
  case 5:
    cubic_add(x, y, z, 1, 0, -1);
    break;
  default:
    assert(false);
  }
}

void axial_add_direction(int &q, int &r, int direction) {
  int x, y, z;
  axial_to_cubic(q, r, x, y, z);
  cubic_add_direction(x, y, z, direction);
  cubic_to_axial(x, y, z, q, r);
}

int cubic_distance(int x0, int y0, int z0, int x1, int y1, int z1) {
  return max(abs(x0 - x1), max(abs(y0 - y1), abs(z0 - z1)));
}

int axial_distance(int q0, int r0, int q1, int r1) {
  int x0, y0, z0;
  int x1, y1, z1;
  axial_to_cubic(q0, r0, x0, y0, z0);
  axial_to_cubic(q1, r1, x1, y1, z1);
  return cubic_distance(x0, y0, z0, x1, y1, z1);
}

const float AGENT_SIZE = 10.0f;
const float EAT_DISTANCE = 16.0f;
const float FOOD_VALUE = 100.0f;
const float MAX_BEHAVIOR_POINTS = 100.0f;
const float MAX_HEALTH_POINTS = 100.0f;
const int DAY_LENGTH = 2000;
const int FORWARD_BLOCKED_DISTANCE_SENSOR_LIMIT = 7;
const int FORWARD_FOOD_DISTANCE_SENSOR_LIMIT = 7;
const int MAX_AGENTS = 200;
const int RECORD_SAMPLE_RATE = 100;
const int RESERVED_AGENT_COUNT = 10;
const int TURBO_RATE = 300;

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> fdis(0, 1);
static std::normal_distribution<float> norm_dist(0, 1);

struct AgentInput {
  float self_behavior_points;
  float self_health_points;
  int forward_blocked_distance;
  int forward_food_distance;
  int time_of_day;
};

struct AgentBehavior {
  bool eating;
  bool spawning;
  // int give_take;
  int linear;
  int rotational;
  bool resting() {
    return !eating && !spawning && linear == 0 && rotational == 0; // &&
    //  give_take == 0;
  }
};

static AgentBehavior agent_behaviors[MAX_AGENTS];

typedef float gene;
static float MAX_GENE_VALUE = FLT_MAX;

struct Agent {
  bool out;
  fann *ann;
  float behavior_points;
  float health_points;
  float hue;
  int parent_index;
  int q, r, orientation;
  int score;

  float mutate_rate0;
  float mutate_rate1;

  gene dna[DNA_SIZE];

  Agent() {
    this->out = true;
    this->ann = fann_create_standard(
        2, ANN_NUM_INPUT, ANN_NUM_OUTPUT); // ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);
  }

  void randomize() {

    for (int i = 0; i < DNA_SIZE; i++) {
      dna[i] = norm_dist(gen);
    }
  }

  void remove_from_world() {
    WorldHex *hex = hex_axial(this->q, this->r);
    if (hex != 0) {
      hex->agent = 0;
    }
    this->out = true;
  }

  void reset_agent() {

    this->gene_cursor = 0;

    this->mutate_rate0 = fabs(next_gene()) / 10.0f;
    this->mutate_rate1 = fabs(next_gene()) / 10.0f;

    fann_connection connections[ANN_NUM_CONNECTIONS];
    fann_get_connection_array(this->ann, connections);
    for (int j = 0; j < ANN_NUM_CONNECTIONS; j++) {
      connections[j].weight = next_gene();
    }
    fann_set_weight_array(this->ann, connections, ANN_NUM_CONNECTIONS);

    fann_set_activation_function_hidden(this->ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(this->ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_steepness_hidden(this->ann, fabs(next_gene()));
    fann_set_activation_steepness_output(this->ann, fabs(next_gene()));
    // fann_set_training_algorithm(this->ann, FANN_TRAIN_INCREMENTAL);
    // fann_set_learning_rate(this->ann, 0.1f);

    this->parent_index = -1;
    this->health_points = MAX_HEALTH_POINTS;
    this->behavior_points = MAX_BEHAVIOR_POINTS;
    this->score = 0;
    this->hue = fabs((float)((int)(next_gene() * 10000.0f) % 10000) / 10000.0f);
    this->out = false;
    WorldHex *hex;
    do {
      this->q = Q * fdis(gen);
      this->r = R * fdis(gen);
      hex = hex_axial(this->q, this->r);
    } while (hex == 0 || hex->agent != 0);
    this->orientation = 6 * fdis(gen);
    hex_axial(this->q, this->r)->agent = this;
  }

  void init_from_parent(int parent_index) {

    for (int i = 0; i < DNA_SIZE; i++) {
      if (fdis(gen) < this->mutate_rate0) {
        this->dna[i] =
            agents[parent_index].dna[i] + (norm_dist(gen) * this->mutate_rate1);
      } else {
        this->dna[i] = agents[parent_index].dna[i];
      }
    }
  }

  static Agent agents[MAX_AGENTS];

  static int select() {
    int total_score = 0;
    for (int i = 0; i < num_agents; i++) {
      Agent &agent = Agent::agents[i];
      if (agent.out) {
        continue;
      }
      total_score += agent.score;
    }

    int selected_index = 0;
    int random_score = (int)(fdis(gen) * (float)total_score);
    for (int i = 0; i < num_agents && random_score >= 0.0f; i++) {
      Agent agent = Agent::agents[i];
      if (agent.out) {
        continue;
      }
      random_score -= agent.score;
      selected_index = i;
    }

    return selected_index;
  }

private:
  int gene_cursor = 0;
  gene next_gene() {
    assert(gene_cursor < DNA_SIZE);
    return this->dna[gene_cursor++];
  }
};

Agent Agent::agents[MAX_AGENTS];

struct Record {
  float hues[MAX_AGENTS];
  gene dna[DNA_SIZE];
  int scores[MAX_AGENTS];
  bool outs[MAX_AGENTS];
  float selected_hue;

  Record() {
    for (int i = 0; i < MAX_AGENTS; i++)
      outs[i] = true;
  }
};

static Record records[WIDTH];

void unit_tests() {
  assert(0 == axial_distance(0, 0, 0, 0));
  assert(1 == axial_distance(0, 0, 1, 0));
  assert(2 == axial_distance(0, 0, 1, 1));
  assert(1 == axial_distance(0, 0, 0, 1));
  assert(10 == axial_distance(0, 0, 10, 0));

  fann *test_ann = fann_create_standard(4, ANN_NUM_INPUT, ANN_NUM_HIDDEN,
                                        ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);
  fann_print_parameters(test_ann);
  printf("actual number of connections: %d (ANN_NUM_CONNECTIONS=%d)\n",
         fann_get_total_connections(test_ann), ANN_NUM_CONNECTIONS);
  assert(fann_get_total_connections(test_ann) == ANN_NUM_CONNECTIONS);
  fann_destroy(test_ann);
}

void init() {
  eg_init(WIDTH, HEIGHT, "Patterns of Life");
  setlocale(LC_NUMERIC, "");
}

void step() {

  // handle user events
  EGEvent event;
  while (eg_poll_event(&event)) {
    switch (event.type) {
    case SDL_QUIT: {
      quit = true;
      break;
    }
    case SDL_MOUSEBUTTONDOWN: {
      SDL_MouseButtonEvent e = event.button;
      break;
    }
    case SDL_KEYDOWN: {
      SDL_KeyboardEvent e = event.key;
      switch (e.keysym.scancode) {
      case SDL_SCANCODE_GRAVE:
        frame_rate = 1;
        if (pause)
          nudge = true;
        else
          pause = true;
        break;
      case SDL_SCANCODE_1:
        frame_rate = 1;
        pause = false;
        break;
      case SDL_SCANCODE_2:
        frame_rate = int((float)TURBO_RATE * 0.04f);
        pause = false;
        break;
      case SDL_SCANCODE_3:
        frame_rate = int((float)TURBO_RATE * 0.20f);
        pause = false;
        break;
      case SDL_SCANCODE_4:
        frame_rate = TURBO_RATE;
        pause = false;
        break;
      case SDL_SCANCODE_5:
        frame_rate = TURBO_RATE * 5.0f;
        pause = false;
        break;
      case SDL_SCANCODE_6:
        frame_rate = TURBO_RATE * 5.0f * 5.0f;
        pause = false;
        break;
      case SDL_SCANCODE_TAB:
        draw_record++;
        nudge = true;
        break;
      case SDL_SCANCODE_LEFTBRACKET:
        --following;
        if (following < 0)
          following = num_agents - 1;
        printf("following=%d\n", following);
        break;
      case SDL_SCANCODE_RIGHTBRACKET:
        ++following;
        if (following >= num_agents)
          following = 0;
        printf("following=%d\n", following);
        break;
      case SDL_SCANCODE_I:
        draw_extra_info = !draw_extra_info;
        nudge = true;
        break;
      case SDL_SCANCODE_C:
        for (int i = 0; i < WORLD_SIZE; i++) {
          world[i].food = false;
        }
        nudge = true;
        break;
      case SDL_SCANCODE_SPACE:
        if (e.repeat == 0) {
          int mouse_x, mouse_y;
          SDL_GetMouseState(&mouse_x, &mouse_y);
          moving_home_x = mouse_x;
          moving_home_y = HEIGHT - mouse_y;
          moving = true;
          following = -1;
        }
        break;
      case SDL_SCANCODE_Z:
        if (e.repeat == 0) {
          int mouse_x, mouse_y;
          SDL_GetMouseState(&mouse_x, &mouse_y);
          zooming_home = HEIGHT - mouse_y;
          zooming = true;
        }
        break;
      case SDL_SCANCODE_MINUS:
        if (e.keysym.mod & KMOD_SHIFT) {
          if (food_spawn_rate > 0.00000001f) {
            food_spawn_rate /= 1.1f;
          }
          printf("food_spawn_rate=%f\n", food_spawn_rate);
        } else {
          if (num_agents > 0) {
            Agent::agents[num_agents - 1].remove_from_world();
            --num_agents;
            printf("num_agents=%d\n", num_agents);
          }
        }
        break;
      case SDL_SCANCODE_EQUALS:
        if (e.keysym.mod & KMOD_SHIFT) {
          if (food_spawn_rate < 1.0f) {
            food_spawn_rate *= 1.1f;
          }
          printf("food_spawn_rate=%f\n", food_spawn_rate);
        } else {
          if (num_agents < MAX_AGENTS) {
            Agent::agents[num_agents].randomize();
            Agent::agents[num_agents++].reset_agent();
            printf("num_agents=%d\n", num_agents);
          }
        }
        break;
      default:
        break;
      }
      break;
    }
    case SDL_KEYUP: {
      SDL_KeyboardEvent e = event.key;
      switch (e.keysym.scancode) {
      case SDL_SCANCODE_SPACE:
        moving = false;
        break;
      case SDL_SCANCODE_Z:
        zooming = false;
        break;
      default:
        break;
      }
    } break;
    }
  }

  if (pause && !nudge)
    return;
  nudge = false;

  // respawn agents 0-n
  for (int i = 0; i < RESERVED_AGENT_COUNT; i++) {
    Agent &agent = Agent::agents[i];
    if (agent.out) {
      agent.randomize();
      agent.reset_agent();
    }
  }

  // spawn food
  if (fdis(gen) < food_spawn_rate) {
    world[(int)(fdis(gen) * WORLD_SIZE)].food = true;
  }

  // day or night
  int time_of_day = frame % DAY_LENGTH;
  bool day = (time_of_day < (DAY_LENGTH / 2));
  float time_of_day_modifier =
      sinf(((float)time_of_day / (float)DAY_LENGTH) * 2.0f * M_PI);

  // map world state to the agent input model
  AgentInput agent_inputs[num_agents];
  for (int i = 0; i < num_agents; i++) {
    Agent agent = Agent::agents[i];
    if (agent.out) {
      continue;
    }
    int x, y, z;
    axial_to_cubic(agent.q, agent.r, x, y, z);
    for (int j = 0; j < FORWARD_FOOD_DISTANCE_SENSOR_LIMIT; j++) {
      agent_inputs[i].forward_food_distance = j;
      WorldHex *hex = hex_cubic(x, y, z);
      if (hex != 0 && hex->food) {
        break;
      }
      cubic_add_direction(x, y, z, agent.orientation);
    }
    axial_to_cubic(agent.q, agent.r, x, y, z);
    for (int j = 0; j < FORWARD_BLOCKED_DISTANCE_SENSOR_LIMIT; j++) {
      agent_inputs[i].forward_blocked_distance = j;
      WorldHex *hex = hex_cubic(x, y, z);
      if (hex == 0 || (i != 0 && hex->agent != 0)) {
        break;
      }
      cubic_add_direction(x, y, z, agent.orientation);
    }
    agent_inputs[i].self_health_points = agent.health_points;
    agent_inputs[i].self_behavior_points = agent.behavior_points;
    agent_inputs[i].time_of_day = time_of_day;
  }

  // behavior model
  for (int i = 0; i < num_agents; i++) {

    Agent &agent = Agent::agents[i];

    if (agent.out) {
      continue;
    }

    // decay health as a function of time
    agent.health_points += -0.1f;

    // death
    if (agent.health_points <= 0.0f) {
      if (following == i && agent.parent_index != -1) {
        following = agent.parent_index;
      }
      for (int j = 0; j < num_agents; j++) {
        Agent &agent = Agent::agents[j];
        if (!agent.out && agent.parent_index == i) {
          agent.parent_index = -1;
          if (following == i) following = j;
        }
      }
      agent.remove_from_world();
      continue;
    }

    float ann_input[ANN_NUM_INPUT];
    ann_input[0] = (float)agent_inputs[i].forward_food_distance /
                   (float)FORWARD_FOOD_DISTANCE_SENSOR_LIMIT;
    ann_input[1] = (float)agent_inputs[i].forward_blocked_distance /
                   (float)FORWARD_BLOCKED_DISTANCE_SENSOR_LIMIT;
    ann_input[2] = agent_inputs[i].self_health_points / MAX_HEALTH_POINTS;
    ann_input[3] = agent_inputs[i].self_behavior_points / MAX_BEHAVIOR_POINTS;
    ann_input[4] = time_of_day_modifier;

    // execute ann
    float *ann_output = fann_run(agent.ann, ann_input);
    agent_behaviors[i].rotational = (int)(ann_output[0] * 1.5f);
    agent_behaviors[i].linear = abs((int)(ann_output[1] * 1.5f));
    agent_behaviors[i].eating = fabs(ann_output[2]) > 0.5f;
    agent_behaviors[i].spawning = fabs(ann_output[3]) > 0.5f;
    // agent_behaviors[i].give_take = lroundf(ann_output[4]);
    if (agent_behaviors[i].resting()) {
      float gain = 10.0f - (time_of_day_modifier * 10.0f);
      agent.behavior_points =
          min(agent.behavior_points + gain, MAX_BEHAVIOR_POINTS);
      continue;
    }

    // apply rotational behavior
    if (fabs(agent_behaviors[i].rotational) < agent.behavior_points) {
      agent.orientation += agent_behaviors[i].rotational;
      while (agent.orientation < 0)
        agent.orientation += 6;
      while (agent.orientation > 5)
        agent.orientation -= 6;
      agent.behavior_points -= fabs(agent_behaviors[i].rotational);
    }

    // apply linear behavior
    hex_axial(agent.q, agent.r)->agent = 0;
    int x, y, z;
    axial_to_cubic(agent.q, agent.r, x, y, z);
    for (int j = 0;
         j < agent_behaviors[i].linear && agent.behavior_points > 1.0f; j++) {
      int x0 = x, y0 = y, z0 = z;
      cubic_add_direction(x0, y0, z0, agent.orientation);
      WorldHex *hex = hex_cubic(x0, y0, z0);
      if (hex != 0 && hex->agent == 0) {
        x = x0;
        y = y0;
        z = z0;
      }
      agent.behavior_points -= 1.0f;
    }
    cubic_to_axial(x, y, z, agent.q, agent.r);
    hex_axial(agent.q, agent.r)->agent = &agent;

    // eat near food
    if (agent_behaviors[i].eating && agent.behavior_points > 1.0f) {
      WorldHex *hex = hex_axial(agent.q, agent.r);
      if (hex != 0 && hex->food) {
        hex->food = false;
        agent.health_points =
            min(MAX_HEALTH_POINTS, agent.health_points + FOOD_VALUE);
        agent.score++;
        agent.behavior_points -= 1.0f;
      }
    }

    // spawning
    if (agent_behaviors[i].spawning && agent.behavior_points > 50.0f &&
        agent.health_points > 50.0f) {
      int new_index = RESERVED_AGENT_COUNT;
      while (new_index < num_agents && !Agent::agents[new_index].out) {
        new_index++;
      }
      if (new_index < num_agents) {
        int new_q = agent.q;
        int new_r = agent.r;
        int random_direction = fdis(gen) * 6.0f;
        axial_add_direction(new_q, new_r, random_direction);
        WorldHex *hex = hex_axial(new_q, new_r);
        if (hex != 0 && hex->agent == 0) {
          Agent::agents[new_index].init_from_parent(i);
          Agent::agents[new_index].reset_agent();
          Agent::agents[new_index].parent_index = i;
          hex_axial(Agent::agents[new_index].q, Agent::agents[new_index].r)
              ->agent = 0;
          Agent::agents[new_index].q = new_q;
          Agent::agents[new_index].r = new_r;
          hex_axial(Agent::agents[new_index].q, Agent::agents[new_index].r)
              ->agent = &Agent::agents[new_index];
          agent.behavior_points = Agent::agents[new_index].behavior_points =
              agent.behavior_points / 2.0f;
          agent.health_points = Agent::agents[new_index].health_points =
              agent.health_points / 2.0f;
        }
      }
    }

    // // give and take
    // if (agent_behaviors[i].give_take != 0 && agent.behavior_points > 1.0f) {
    //   int target_q = agent.q;
    //   int target_r = agent.r;
    //   axial_add_direction(target_q, target_r, agent.orientation);
    //   WorldHex *hex = hex_axial(target_q, target_r);
    //   if (hex != 0 && hex->agent != 0) {
    //     Agent *target = hex->agent;
    //     target->health_points += (float)agent_behaviors[i].give_take;
    //     agent.health_points -= (float)agent_behaviors[i].give_take;
    //     agent.behavior_points -= 1.0f;
    //   }
    // }
  }

  // update the record model
  if (frame % RECORD_SAMPLE_RATE == 0 && num_agents > 0) {

    int selected_index = Agent::select();

    records[records_index].selected_hue = Agent::agents[selected_index].hue;

    for (int i = 0; i < DNA_SIZE; i++) {
      records[records_index].dna[i] =
          Agent::agents[selected_index].dna[i]; // memcpy
    }

    for (int i = 0; i < MAX_AGENTS; i++) {
      const Agent &agent = Agent::agents[i];
      records[records_index].scores[i] = agent.score;
      records[records_index].hues[i] = agent.hue;
      records[records_index].outs[i] = agent.out;
    }

    records_index++;
    records_index %= WIDTH;
  }

  // display
  //
  //
  //
  if (frame % frame_rate == 0 || moving || zooming) {

    eg_clear_screen(0.0f, 0.0f, 0.0f, 0.0f);
    eg_reset_transform();

    if (draw_record % 4 == 0) {

      if (moving) {
        int mouse_x, mouse_y;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        mouse_y = HEIGHT - mouse_y;
        camera_x -= (float)(mouse_x - moving_home_x) * (1.0f / camera_zoom);
        camera_y -= (float)(mouse_y - moving_home_y) * (1.0f / camera_zoom);
        // warp_mouse(WIDTH / 2, HEIGHT / 2);
        moving_home_x = mouse_x;
        moving_home_y = mouse_y;
      }

      if (zooming) {
        int mouse_x, mouse_y;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        mouse_y = HEIGHT - mouse_y;
        camera_zoom += (float)(mouse_y - zooming_home) * 0.003f;
        // warp_mouse(WIDTH / 2, HEIGHT / 2);
        zooming_home = mouse_y;
      }

      if (following != -1) {
        int x, y;
        axial_to_xy(Agent::agents[following].q, Agent::agents[following].r, x,
                    y);
        camera_x = x;
        camera_y = y;
      }

      eg_scale(camera_zoom, camera_zoom);
      eg_translate(-camera_x, -camera_y);
      eg_translate((float)(WIDTH / 2) / camera_zoom,
                   (float)(HEIGHT / 2) / camera_zoom);

      int border_x0, border_y0, border_x1, border_y1, border_x2, border_y2,
          border_x3, border_y3;
      axial_to_xy(0, 0, border_x0, border_y0);
      axial_to_xy(Q - 1, 0, border_x1, border_y1);
      axial_to_xy(Q - 1, R - 1, border_x2, border_y2);
      axial_to_xy(0, R - 1, border_x3, border_y3);

      eg_set_color(1, 1, 1, 0.1f);
      glLineWidth(1);
      glBegin(GL_POLYGON);
      glVertex2f(border_x0, border_y0);
      glVertex2f(border_x1, border_y1);
      glVertex2f(border_x2, border_y2);
      glVertex2f(border_x3, border_y3);
      glEnd();

      // draw foods
      for (int q = 0; q < Q; q++) {
        for (int r = 0; r < R; r++) {
          if (hex_axial(q, r)->food) {
            eg_push_transform();
            int x, y;
            axial_to_xy(q, r, x, y);
            eg_translate(x, y);
            eg_scale(HEX_SIZE, HEX_SIZE);
            eg_scale(0.5f, 0.5f);
            eg_set_color(0.1f, 0.9f, 0.1f, 1.0f);
            eg_draw_square(-0.5f, -0.5, 1, 1);
            eg_pop_transform();
          }
        }
      }

      // draw agents
      for (int i = 0; i < num_agents; i++) {
        Agent agent = Agent::agents[i];
        if (agent.out) {
          continue;
        }

        // pixel location
        int x, y;
        axial_to_xy(agent.q, agent.r, x, y);

        // indicate orientation
        float r, g, b;
        hsv_to_rgb(agent.hue, 1.0f, 1.0f, &r, &g, &b);
        eg_set_color(r, g, b, 1.0f);
        float angle = agent.orientation / 6.0f * 2 * M_PI + (M_PI / 6.0f);
        float orientation_line_length = 15.0;
        // agent_behaviors[i].give_take != 0 ? 30.0f : 10.0f;
        eg_draw_line(x, y, x + (float)cos(angle) * orientation_line_length,
                     y + (float)sin(angle) * orientation_line_length,
                     //  10.0f);
                     agent.parent_index == -1 ? 10.0f : 5.0f);

        // agent
        eg_push_transform();
        eg_translate(x, y);
        eg_rotate((agent.orientation / 6.0f) * 360.0f + (360 / 12));
        float buddy_size = // 1.0f;
            AGENT_SIZE * (agent.parent_index == -1 ? 1.0f : 0.6f);
        eg_scale(buddy_size, buddy_size);
        eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
        eg_draw_square(-0.5f, -0.5f, 1.0f, 1.0f);
        eg_scale(0.5f, 0.5f);
        eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
        if (agent_behaviors[i].eating) {
          eg_set_color(0.3f, 0.9f, 0.2f, 1.0f);
        }
        eg_draw_square(-0.5f, -0.5f, 1.0f, 1.0f);
        eg_pop_transform();

        if (draw_extra_info) {
          // health bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(x - 15.0f, y + 12.0f, 30.0f, 5.0f);
          if (agent.health_points > MAX_HEALTH_POINTS * 0.25f) {
            eg_set_color(0.5f, 0.9f, 0.5f, 0.8f);
          } else {
            eg_set_color(0.8f, 0.3f, 0.3f, 0.8f);
          }
          eg_draw_square(x - 15.0f, y + 12.0f,
                         agent.health_points * 30.0f / MAX_HEALTH_POINTS, 5.0f);

          // behavior points bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(x - 15.0f, y + 20.0f, 30.0f, 5.0f);
          eg_set_color(0.10f, 0.10f, 0.9f,
                       agent_behaviors[i].resting() ? 1.0f : 0.7f);
          eg_draw_square(x - 15.0f, y + 20.0f,
                         30.0f * (agent.behavior_points / MAX_BEHAVIOR_POINTS),
                         5.0f);

          // score bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(x - 15.0f, y + 28.0f, 30.0f, 5.0f);
          eg_set_color(0.9f, 0.85f, 0.0f, 0.8f);
          eg_draw_square(
              x - 15.0f, y + 28.0f,
              30.0f * min(1.0f, ((float)agent.score / (float)HEIGHT)), 5.0f);
        }
      }

      // draw day or night cast
      if (frame_rate == 1) {
        eg_push_transform();
        eg_reset_transform();
        if (day) {
          eg_set_color(1.0f, 1.0f, 0.0f, 0.1f * fabs(time_of_day_modifier));
        } else {
          eg_set_color(0.0f, 0.0f, 1.0f, 0.1f * fabs(time_of_day_modifier));
        }
        eg_draw_square(0, 0, WIDTH, HEIGHT);
        eg_pop_transform();
      }
    }

    // gene graph
    if (draw_record % 4 == 1) {
      for (int rx = 0; rx < WIDTH; rx++) {
        const Record &record = records[(records_index + rx) % WIDTH];
        float y = 0.0f;
        float total = 0.0f;
        for (int wi = 0; wi < DNA_SIZE; wi++) {
          total += fabs(record.dna[wi]);
        }
        for (int wi = 0; wi < DNA_SIZE; wi++) {
          float r, g, b;
          hsv_to_rgb(record.selected_hue, 1.0, wi % 2 == 0 ? 0.0 : 1.0, &r, &g,
                     &b);
          eg_set_color(r, g, b, 1.0f);
          float h = (fabs(record.dna[wi]) / total) * HEIGHT * 2;
          eg_draw_line(rx, y, rx, y + h, 1.5f);
          y += h;
        }
      }
    }

    // total score graph
    if (draw_record % 4 == 2) {
      for (int rx = 0; rx < WIDTH; rx++) {
        const Record &record = records[(records_index + rx) % WIDTH];
        for (int i = 0; i < MAX_AGENTS; ++i) {
          if (!record.outs[i]) {
            float r, g, b;
            hsv_to_rgb(record.hues[i], 1.00f, 1.00f, &r, &g, &b);
            eg_set_color(r, g, b, 1.0f);
            eg_draw_square(rx, record.scores[i] % HEIGHT, 1.0f, 1.0f);
          }
        }
      }
    }

    // population graph
    if (draw_record % 4 == 3) {
      float h = (float)HEIGHT / (float)MAX_AGENTS;
      for (int rx = 0; rx < WIDTH; rx++) {
        const Record &record = records[(records_index + rx) % WIDTH];
        float y = 0;
        for (int i = 0; i < MAX_AGENTS; ++i) {
          float r, g, b;
          hsv_to_rgb(record.hues[i], 1.00f, record.outs[i] ? 0.5f : 1.0f, &r,
                     &g, &b);
          eg_set_color(r, g, b, 1.0f);
          eg_draw_line(rx, y, rx, y + h, 1.0f);
          y += h;
        }
      }
    }

    eg_swap_buffers();
  }

  frame++;
}

int main(int argc, char *argv[]) {

  unit_tests();
  init();

#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop(step, 0, 1);
#else
  while (!quit) {
    step();
  }
#endif

  printf("frames=%'d\ndays=%'d\nyears=%'d\n", frame, frame / DAY_LENGTH,
         frame / DAY_LENGTH / 365);

  eg_shutdown();

  return 0;
}
