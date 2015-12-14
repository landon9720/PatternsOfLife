#include "patterns.h"

class Node {
  public:
    Node(int input_count, float **inputs, float **weights) {
      this->input_count = input_count;
      this->inputs = inputs;
      this->weights = weights;
    }
    float activate() {
      float sum = 0.0f;
      for (int i = 0; i < input_count; i++) {
        sum += *(inputs[i]) * *(weights[i]);
      }
      value = activation_function(sum);
      return value;
    }
    float value;
  private:
    float activation_function(float v) {
      return tanh(v);
    }
    int input_count;
    float **inputs;
    float **weights;
};

struct Agent;

class Sensor {
public:
  virtual float sense(const Agent &agent) = 0;
};

class Behavior {
public:
  virtual bool behave(Agent &agent, float perceptron_output) = 0;
};

struct WorldHex {
  bool food;
  Agent *agent;
};

const int HEX_SIZE = 15;
const int WIDTH = 1280 * 0.9f;
const int HEIGHT = 720 * 0.9f;
const int Q = 100;
const int R = 100;
const int WORLD_SIZE = Q * R;

const int DNA_SIZE = 12;

static bool draw_extra_info = false;
static bool moving = false;
static bool nudge = false;
static bool paused = false;
static bool quit = false;
static bool zooming = false;
static float camera_x = HEX_SIZE * R;
static float camera_y = HEX_SIZE * Q;
static float camera_zoom = 0.5f;
static float food_spawn_rate = 0.304482f;
// static float time_of_day_modifier = 0.0f;
static int draw_record = 0;
static int following = -1;
static int frame = 0;
static int frame_rate = 1;
static int moving_home_x;
static int moving_home_y;
static int num_agents = 50;
static int records_index = 0;
// static int time_of_day = 0;
static int zooming_home;

void cubic_to_axial(int x, int y, int z, int &q, int &r) {
  q = x;
  r = z;
}

void axial_to_cubic(int q, int r, int &x, int &y, int &z) {
  x = q;
  z = r;
  y = -x-z;
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

float ssqrtf3 = sqrtf(3.0f);

void axial_to_xy(int q, int r, int &x, int &y) {
  x = HEX_SIZE * 3.0f / 2.0f * q;
  y = HEX_SIZE * ssqrtf3 * (r + q / 2.0f);
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

int direction_add(int direction, int rotation) {
  assert(direction >= 0);
  assert( direction <= 5);
  int result = direction + rotation;
  while (result < 0)
    result += 6;
  while (result > 5)
    result -= 6;
  return result;
}

// int cubic_distance(int x0, int y0, int z0, int x1, int y1, int z1) {
//   return max(abs(x0 - x1), max(abs(y0 - y1), abs(z0 - z1)));
// }

// int axial_distance(int q0, int r0, int q1, int r1) {
//   int x0, y0, z0;
//   int x1, y1, z1;
//   axial_to_cubic(q0, r0, x0, y0, z0);
//   axial_to_cubic(q1, r1, x1, y1, z1);
//   return cubic_distance(x0, y0, z0, x1, y1, z1);
// }

const float AGENT_SIZE = 20.0f;
const float FOOD_VALUE = 100.0f;
// const float MAX_BEHAVIOR_POINTS = 100.0f;
const float MAX_HEALTH_POINTS = 100.0f;
const int DAY_LENGTH = 2000;
// const int FORWARD_BLOCKED_DISTANCE_SENSOR_LIMIT = 7;
// const int FORWARD_FOOD_DISTANCE_SENSOR_LIMIT = 7;
const int MAX_AGENTS = 1000;
const int RECORD_SAMPLE_RATE = 100;
const int RESERVED_AGENT_COUNT = 20;
const int TURBO_RATE = 307;

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> fdis(0, 1);
static std::normal_distribution<float> norm_dist(0, 1);

typedef float gene;

struct Agent {
  bool out;
  // float behavior_points;
  float health_points;
  float hue;
  int q, r, orientation;
  int score;

  gene dna[DNA_SIZE];

  Agent() {
    this->out = true;
  }

  void randomize() {
    for (int i = 0; i < DNA_SIZE; i++) {
      dna[i] = norm_dist(gen);
    }
  }

  void remove_from_world() {
    WorldHex *hex = hex_axial(this->q, this->r);
    assert(hex != 0);
    hex->agent = 0;
    this->out = true;
  }

  void reset_agent() {

    this->gene_cursor = 0;

    this->health_points = MAX_HEALTH_POINTS;
    // this->behavior_points = MAX_BEHAVIOR_POINTS;
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

  void init_from_parent(Agent *parent) {
    for (int i = 0; i < DNA_SIZE; i++) {
      this->dna[i] = parent->dna[i];
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

// static const int NONE_MARK = 0;
// static const int DEATH_MARK = 1;
// static const int BIRTH_MARK = 2;
// static const int EAT_MARK = 3;

// struct Mark {
//   int type;
//   int q, r;
//   int frame;
// };

// static const int NUM_MARKS = MAX_AGENTS * 3;
// static Mark marks[NUM_MARKS];

// static int mark_cursor = 0;

// void add_mark(const Mark &mark) {
//   marks[mark_cursor++] = mark;
//   mark_cursor %= NUM_MARKS;
// }

class FoodSensor : public Sensor {
public:
  FoodSensor(int relative_direction, int distance) {
    this->relative_direction = relative_direction;
    this->distance = distance;
  }
  virtual float sense(const Agent &agent) {
    int direction = direction_add(agent.orientation, relative_direction);
    int x, y, z;
    axial_to_cubic(agent.q, agent.r, x, y, z);
    for (int j = 0; j < distance; j++) {
      cubic_add_direction(x, y, z, direction);
    }
    WorldHex *hex = hex_cubic(x, y, z);
    if (hex != 0 && hex->food) {
      return 1.0f;
    } else {
      return 0.0f;
    }
    // for (int j = 0; j < FORWARD_FOOD_DISTANCE_SENSOR_LIMIT; j++) {
    //   WorldHex *hex = hex_cubic(x, y, z);
    //   if (hex != 0 && hex->food) {
    //     return (FORWARD_FOOD_DISTANCE_SENSOR_LIMIT - (float)j) / (float)FORWARD_FOOD_DISTANCE_SENSOR_LIMIT;
    //   }
    //   cubic_add_direction(x, y, z, direction);
    // }
    // return 1.0f;
  }

private:
  int relative_direction;
  int distance;
};

// class ForwardBlockedSensor : public Sensor {
// public:
//   virtual float sense(const Agent &agent) {
//     int x, y, z;
//     axial_to_cubic(agent.q, agent.r, x, y, z);
//     for (int j = 0; j < FORWARD_BLOCKED_DISTANCE_SENSOR_LIMIT; j++) {
//       WorldHex *hex = hex_cubic(x, y, z);
//       if (hex == 0 || (j != 0 && hex->agent != 0)) {
//         return (float)j / (float)FORWARD_BLOCKED_DISTANCE_SENSOR_LIMIT;
//       }
//       cubic_add_direction(x, y, z, agent.orientation);
//     }
//     return 1.0f;
//   }
// };

// class SelfHealthPointsSensor : public Sensor {
// public:
//   virtual float sense(const Agent &agent) {
//     return (float)agent.health_points / (float)MAX_HEALTH_POINTS;
//   }
// };

// class SelfBehaviorPointsSensor : public Sensor {
// public:
//   virtual float sense(const Agent &agent) {
//     return (float)agent.behavior_points / (float)MAX_BEHAVIOR_POINTS;
//   }
// };

// class TimeOfDaySensor : public Sensor {
// public:
//   virtual float sense(const Agent &agent) { return time_of_day_modifier; }
// };

// class AltSensor : public Sensor {
// public:
//   virtual float sense(const Agent &agent) {
//     return hex_axial(agent.q, agent.r)->alt;
//   }
// };

class RotationalBehavior : public Behavior {
public:
  virtual bool behave(Agent &agent, float perceptron_output) {
    int rotational = 0; 
    if (perceptron_output < 0.5f)
      rotational = -1;
    else if (perceptron_output > 0.5f)
      rotational = 1;
    else
      return false;
    // if (rotational == 0)
      // return false;
    // if (agent.behavior_points > 1.0f) {
      agent.orientation = direction_add(agent.orientation, rotational);
      // agent.behavior_points -= fabs((float)rotational);
    // }
    return true;
  }
};

class LinearBehavior : public Behavior {
public:
  virtual bool behave(Agent &agent, float perceptron_output) {
    int linear = 1;//abs((int)(perceptron_output * 1.5f));
    // if (linear == 0)
    //   return false;
    hex_axial(agent.q, agent.r)->agent = 0;
    int x, y, z;
    axial_to_cubic(agent.q, agent.r, x, y, z);
    for (int j = 0; j < linear; j++) {
      // float alt = hex_cubic(x, y, z)->alt;
      // float cost = 2.0f;
      // if (alt > 0.5f) {
        // cost += alt * 50.0f;
      // }
      // if (agent.behavior_points < cost) {
      //   break;
      // }
      int x0 = x, y0 = y, z0 = z;
      cubic_add_direction(x0, y0, z0, agent.orientation);
      WorldHex *hex = hex_cubic(x0, y0, z0);
      if (hex != 0 && hex->agent == 0) {
        x = x0;
        y = y0;
        z = z0;
        // agent.behavior_points -= cost;
      }
    }
    cubic_to_axial(x, y, z, agent.q, agent.r);
    hex_axial(agent.q, agent.r)->agent = &agent;
    return true;
  }
};

class EatingBehavior : public Behavior {
public:
  virtual bool behave(Agent &agent, float perceptron_output) {
    // bool eating = true;//fabs(perceptron_output) > 0.5f;
    // if (!eating)
      // return false;
    // float cost = 10.0f;//fabs((MAX_HEALTH_POINTS - agent.health_points) - FOOD_VALUE);
    // if (eating) { // && agent.behavior_points > cost) {
      WorldHex *hex = hex_axial(agent.q, agent.r);
      if (hex != 0 && hex->food) {
        hex->food = false;
        agent.health_points =
            min(MAX_HEALTH_POINTS, agent.health_points + FOOD_VALUE);
        agent.score++;
        // agent.behavior_points -= cost;
        // add_mark((struct Mark){EAT_MARK, agent.q, agent.r, frame});
        }
    // }
    return true;
  }
};

class SpawningBehavior : public Behavior {
public:
  virtual bool behave(Agent &agent, float perceptron_output) {
    // bool spawning = true;//fabs(perceptron_output) > 0.5f;
    // if (!spawning)
      // return false;
    // if (agent.health_points > (MAX_HEALTH_POINTS / 2.0f)) { // agent.behavior_points > 50.0f && 
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
          Agent::agents[new_index].init_from_parent(&agent);
          Agent::agents[new_index].reset_agent();
          // Agent::agents[new_index].parent = &agent;
          hex_axial(Agent::agents[new_index].q, Agent::agents[new_index].r)->agent = 0;
          Agent::agents[new_index].q = new_q;
          Agent::agents[new_index].r = new_r;
          hex_axial(Agent::agents[new_index].q, Agent::agents[new_index].r)->agent = &Agent::agents[new_index];
          // agent.behavior_points = Agent::agents[new_index].behavior_points =
          //     agent.behavior_points / 2.0f;
          agent.health_points = Agent::agents[new_index].health_points =
              agent.health_points / 2.0f;
          // add_mark((struct Mark){BIRTH_MARK, new_q, new_r, frame});
        }
      }
    // }
    return true;
  }
};

FoodSensor foodSensor_here(0, 0);
FoodSensor foodSensor_ahead1(0, 1);
// FoodSensor foodSensor_ahead2(0, 2);
// FoodSensor foodSensor_ahead3(0, 3);
// FoodSensor foodSensor1(1);
// FoodSensor foodSensor2(2);
// FoodSensor foodSensor3(3);
// FoodSensor foodSensor4(4);
// FoodSensor foodSensor5(5);
// ForwardBlockedSensor forwardBlockedSensor;
// SelfHealthPointsSensor selfHealthPointsSensor;
// SelfBehaviorPointsSensor selfBehaviorPointsSensor;
// TimeOfDaySensor timeOfDaySensor;
// AltSensor altSensor;

RotationalBehavior rotationalBehavior;
LinearBehavior linearBehavior;
EatingBehavior eatingBehavior;
SpawningBehavior spawningBehavior;

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
        if (paused)
          nudge = true;
        else
          paused = true;
        break;
      case SDL_SCANCODE_1:
        frame_rate = 1;
        paused = false;
        break;
      case SDL_SCANCODE_2:
        frame_rate = int((float)TURBO_RATE * 0.04f);
        paused = false;
        break;
      case SDL_SCANCODE_3:
        frame_rate = int((float)TURBO_RATE * 0.20f);
        paused = false;
        break;
      case SDL_SCANCODE_4:
        frame_rate = TURBO_RATE;
        paused = false;
        break;
      case SDL_SCANCODE_5:
        frame_rate = TURBO_RATE * 5.0f;
        paused = false;
        break;
      case SDL_SCANCODE_6:
        frame_rate = TURBO_RATE * 5.0f * 5.0f;
        paused = false;
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

  if (paused && !nudge)
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
      agent.remove_from_world();
      continue;
    }
    
    float w = 0.5f;
    
    float input1 = foodSensor_here.sense(agent);
    float input2 = foodSensor_ahead1.sense(agent);
    
    agent.gene_cursor = 0;
    float w1 = agent.next_gene();
    float w2 = agent.next_gene();
    float w3 = agent.next_gene();
    float w4 = agent.next_gene();
    float w5 = agent.next_gene();
    float w6 = agent.next_gene();
    float w7 = agent.next_gene();
    float w8 = agent.next_gene();
    
    float *inputs[2] = { &input1, &input2 };
    
    float *weights1[2] = { &w1, &w2};
    float *weights2[2] = { &w3, &w4};
    float *weights3[2] = { &w5, &w6};
    float *weights4[2] = { &w7, &w8};
    
    Node node1 = Node(1, inputs, weights1);
    Node node2 = Node(1, inputs, weights2);
    Node node3 = Node(1, inputs, weights3);
    Node node4 = Node(1, inputs, weights4);
    
    if (node1.activate() > agent.next_gene()) {
      eatingBehavior.behave(agent, 1.0f);
    }
    
    if (node2.activate() > agent.next_gene()) {
      linearBehavior.behave(agent, 1.0f);
    }
    
    rotationalBehavior.behave(agent, node3.activate() * agent.next_gene());
    
    if (node4.activate() > agent.next_gene()) {
      if (agent.health_points > (MAX_HEALTH_POINTS / 2.0f)) {
        spawningBehavior.behave(agent, 1.0f);
      }
    }
  }
    
  // update record model
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
        moving_home_x = mouse_x;
        moving_home_y = mouse_y;
      }

      if (zooming) {
        int mouse_x, mouse_y;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        mouse_y = HEIGHT - mouse_y;
        camera_zoom += (float)(mouse_y - zooming_home) * 0.003f;
        zooming_home = mouse_y;
      }

      if (following != -1) {
        int x, y;
        axial_to_xy(Agent::agents[following].q, Agent::agents[following].r, x, y);
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

      for (int q = 0; q < Q; q++) {
        for (int r = 0; r < R; r++) {
          WorldHex *hex = hex_axial(q, r);

          eg_push_transform();
          int x, y;
          axial_to_xy(q, r, x, y);
          eg_translate(x, y);
          eg_scale(HEX_SIZE, HEX_SIZE);

          // draw foods
          if (hex->food) {
            eg_scale(0.5f, 0.5f);
            eg_set_color(0.05f, 0.8f, 0.05f, 1.0f);
            eg_draw_square(-0.5f, -0.5, 1, 1);
          }

          eg_pop_transform();
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
        float orientation_line_length = 20.0;
        eg_draw_line(x, y, x + (float)cos(angle) * orientation_line_length,
                     y + (float)sin(angle) * orientation_line_length,
                      10.0f);

        // agent
        eg_push_transform();
        eg_translate(x, y);
        eg_rotate((agent.orientation / 6.0f) * 360.0f + (360 / 12));
        float buddy_size = AGENT_SIZE * 1.0f;
        eg_scale(buddy_size, buddy_size);
        eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
        eg_draw_square(-0.5f, -0.5f, 1.0f, 1.0f);
        eg_scale(0.5f, 0.5f);
        eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
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
        }
      }

      // // draw day or night cast
      // if (frame_rate == 1) {
      //   eg_push_transform();
      //   eg_reset_transform();
      //   if (day) {
      //     eg_set_color(1.0f, 1.0f, 0.0f, 0.1f * fabs(time_of_day_modifier));
      //   } else {
      //     eg_set_color(0.0f, 0.0f, 1.0f, 0.1f * fabs(time_of_day_modifier));
      //   }
      //   eg_draw_square(0, 0, WIDTH, HEIGHT);
      //   eg_pop_transform();
      // }

      // draw annotations (marks)
    //   for (int i = 0; i < NUM_MARKS; i++) {
    //     Mark &mark = marks[i];
    //     int age = frame - mark.frame;
    //     switch (mark.type) {
    //     case NONE_MARK:
    //       continue;
    //     case DEATH_MARK:
    //       if (age < 2000) {
    //         eg_push_transform();
    //         int x, y;
    //         axial_to_xy(mark.q, mark.r, x, y);
    //         eg_translate(x, y);
    //         eg_set_color(1, 0.2f, 0.2f, 1 - (age / 2000.0f));
    //         eg_draw_line(-5, -5, 5, 5, 5);
    //         eg_draw_line(5, -5, -5, 5, 5);
    //         eg_pop_transform();
    //       }
    //       break;
    //     case BIRTH_MARK:
    //       if (age < 2000) {
    //         eg_push_transform();
    //         int x, y;
    //         axial_to_xy(mark.q, mark.r, x, y);
    //         eg_translate(x, y);
    //         eg_set_color(0.2f, 0.2f, 1, 1 - (age / 2000.0f));
    //         eg_draw_line(0, 6, 0, -6, 6);
    //         eg_draw_line(-6, 0, 6, 0, 6);
    //         eg_pop_transform();
    //       }
    //       break;
    //     case EAT_MARK:
    //       if (age < 100) {
    //         eg_push_transform();
    //         int x, y;
    //         axial_to_xy(mark.q, mark.r, x, y);
    //         eg_translate(x, y);
    //         eg_set_color(0.10f, 0.7f, 0.05f, 1.0f);
    //         eg_rotate((age / 100.0f) * (4 * 360));
    //         eg_draw_square(-4, -4, 8, 8);
    //         eg_pop_transform();
    //       }
    //       break;
    //     default:
    //       assert(false);
    //     }
    //   }
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
      float h = (float)HEIGHT / (float)num_agents;
      for (int rx = 0; rx < WIDTH; rx++) {
        const Record &record = records[(records_index + rx) % WIDTH];
        float y = 0;
        for (int i = 0; i < num_agents; ++i) {
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

void unit_tests();

int main(int argc, char *argv[]) {
  unit_tests();
  init();
  while (!quit) {
    step();
  }
  printf("frames=%'d\ndays=%'d\nyears=%'d\n", frame, frame / DAY_LENGTH, frame / DAY_LENGTH / 365);
  eg_shutdown();
  return 0;
}

void unit_tests() {

  int x = 0, y = 0, z = 0, q = 0, r = 0;
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  cubic_to_axial(x, y, z, q, r);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  
  cubic_add_direction(x, y, z, 0);
  cubic_add_direction(x, y, z, 3);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  
  cubic_add_direction(x, y, z, 1);
  cubic_add_direction(x, y, z, 4);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  
  cubic_add_direction(x, y, z, 2);
  cubic_add_direction(x, y, z, 5);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  
  cubic_add_direction(x, y, z, 3);
  cubic_add_direction(x, y, z, 0);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  
  axial_add_direction(q, r, 0);
  axial_add_direction(q, r, 3);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
    
  axial_add_direction(q, r, 1);
  axial_add_direction(q, r, 4);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
    
  axial_add_direction(q, r, 2);
  axial_add_direction(q, r, 5);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
      
  axial_add_direction(q, r, 3);
  axial_add_direction(q, r, 0);
  assert(x == 0 && y == 0 && z == 0 && q == 0 && r == 0);
  
  
  // void cubic_to_axial(int x, int y, int z, int &q, int &r) {
  // void axial_to_cubic(int q, int r, int &x, int &y, int &z) {
  // WorldHex *hex_axial(int q, int r)
  // WorldHex *hex_cubic(int x, int y, int z)
  // void axial_to_xy(int q, int r, int &x, int &y)
  // void cubic_add(int &x, int &y, int &z, int dx, int dy, int dz)
  // void cubic_add_direction(int &x, int &y, int &z, int direction) {
  // void axial_add_direction(int &q, int &r, int direction) {
  // direction_add(int direction, int rotation)
  // cubic_distance(int x0, int y0, int z0, int x1, int y1, int z1)
  // axial_distance(int q0, int r0, int q1, int r1)

}
