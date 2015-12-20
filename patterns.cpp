#include "patterns.h"

int num_agents = 0;
float agent_spawn_rate = 0.0f;
float food_spawn_rate = 0.0f;
float food_value = 0.0f;
float max_hp = 0.0f;
int rotational_waiting = 0;
int linear_waiting = 0;
int eating_waiting = 0;
int kill_waiting = 0;
int spawning_waiting = 0;
int birth_waiting = 0;
float burn_rate = 0.0f;
int turbo_rate = 0;

struct Agent;

class Sensor {
public:
  virtual float sense(const Agent &agent) = 0;
};

class Behavior {
public:
  virtual void behave(Agent &agent, float perceptron_output) = 0;
};

struct WorldHex {
  char food;
  Agent *agent;
};

const int HEX_SIZE = 50;
const int WIDTH = 1280;
const int HEIGHT = 800;
const int Q = 120;
const int R = 120;
const int WORLD_SIZE = Q * R;

const int DNA_SIZE = 14 * 8 + 8 * 9 + 9;
const int MEMORY_SIZE = 4;

static bool draw_extra_info = false;
static bool moving = false;
static bool nudge = false;
static bool paused = false;
static bool quit = false;
static bool zooming = false;
static float camera_x = HEX_SIZE * R;
static float camera_y = HEX_SIZE * Q;
static float camera_zoom = 1.0f;
static int draw_record = 0;
static int following = -1;
static int frame = 0;
static int frame_rate = 1;
static int moving_home_x;
static int moving_home_y;
static int records_index = 0;
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

const int DAY_LENGTH = 2000;
const int max_agents = 2000;
const int RECORD_SAMPLE_RATE = 100;

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> fdis(0, 1);
static std::normal_distribution<float> norm_dist(0, 1);

struct Agent {
  bool out;
  float health_points;
  float hue;
  int q, r, orientation;
  int score;
  int waiting;
  
  float memory[MEMORY_SIZE];
  float dna[DNA_SIZE];

  Agent() {
    this->out = true;
  }

  void randomize() {
    for (int i = 0; i < DNA_SIZE; i++) {
      dna[i] = norm_dist(gen);
    }
    this->hue = fabs((float)((int)(fdis(gen) * 10000.0f) % 10000) / 10000.0f);
  }

  void reset_agent() {
    this->health_points = max_hp;
    this->score = 0;
    this->out = false;
    this->waiting = 0;
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
    for (int i = 0; i < DNA_SIZE-1; i++) {
      if (fdis(gen) < 0.1f) {
        this->dna[i] = parent->dna[i] + (norm_dist(gen) * 0.2f);  
      } else {
        this->dna[i] = parent->dna[i];
      }
    }
    this->hue = parent->hue;
  }
};

Agent agents[max_agents];

int select() {
  int total_score = 0;
  for (int i = 0; i < num_agents; i++) {
    Agent &agent = agents[i];
    if (agent.out) {
      continue;
    }
    total_score += agent.score;
  }
  int selected_index = 0;
  int random_score = (int)(fdis(gen) * (float)total_score);
  for (int i = 0; i < num_agents && random_score >= 0.0f; i++) {
    Agent agent = agents[i];
    if (agent.out) {
      continue;
    }
    random_score -= agent.score;
    selected_index = i;
  }
  return selected_index;
}

void remove_from_world(Agent &agent) {
  if (!agent.out) {
    WorldHex *hex = hex_axial(agent.q, agent.r);
    assert(hex != 0);
    assert(hex->agent == &agent);
    hex->agent = 0;
    agent.out = true;
  }
}

struct Record {
  float hues[max_agents];
  float dna[DNA_SIZE];
  int scores[max_agents];
  bool outs[max_agents];
  float selected_hue;

  Record() {
    for (int i = 0; i < max_agents; i++)
      outs[i] = true;
  }
};

static Record records[WIDTH];

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
    if (hex != 0 && hex->food > 0) {
      return 1.0f;
    } else {
      return 0.0f;
    }
  }

private:
  int relative_direction;
  int distance;
};

class AgentSensor : public Sensor {
public:
  AgentSensor(int relative_direction, int distance) {
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
    if (hex != 0 && hex->agent) {
      return hex->agent->hue;
    } else {
      return 0.0f;
    }
  }

private:
  int relative_direction;
  int distance;
};

class SelfHealthPointsSensor : public Sensor {
public:
  virtual float sense(const Agent &agent) {
    return (float)agent.health_points / (float)max_hp;
  }
};

class RotationalBehavior : public Behavior {
public:
  virtual void behave(Agent &agent, float perceptron_output) { 
    if (perceptron_output < 0.5f) {
      agent.orientation = direction_add(agent.orientation, +1);
      agent.waiting += rotational_waiting;
    } else if (perceptron_output > 0.5f) {
      agent.orientation = direction_add(agent.orientation, -1);
      agent.waiting += rotational_waiting;
    }
  }
};

class LinearBehavior : public Behavior {
public:
  virtual void behave(Agent &agent, float perceptron_output) {
    hex_axial(agent.q, agent.r)->agent = 0;
    int x, y, z;
    axial_to_cubic(agent.q, agent.r, x, y, z);
    int x0 = x, y0 = y, z0 = z;
    cubic_add_direction(x0, y0, z0, agent.orientation);
    WorldHex *hex = hex_cubic(x0, y0, z0);
    if (hex != 0 && hex->agent == 0) {
      x = x0;
      y = y0;
      z = z0;
    }
    cubic_to_axial(x, y, z, agent.q, agent.r);
    hex_axial(agent.q, agent.r)->agent = &agent;
    agent.waiting += linear_waiting;
  }
};

class KillBehavior : public Behavior {
public:
  virtual void behave(Agent &agent, float perceptron_output) {
    int x, y, z;
    axial_to_cubic(agent.q, agent.r, x, y, z);
    int targetx = x, targety = y, targetz = z;
    cubic_add_direction(targetx, targety, targetz, agent.orientation);
    WorldHex *hex = hex_cubic(targetx, targety, targetz);
    if (hex != 0 && hex->agent) {
      Agent *target = hex->agent;
      remove_from_world(*target);
    //   hex->food |= 2;
      agent.waiting += kill_waiting;
    }
  }
};

class EatingBehavior : public Behavior {
public:
  virtual void behave(Agent &agent, float perceptron_output) {
    WorldHex *hex = hex_axial(agent.q, agent.r);
    if (hex != 0 && hex->food > 0) {
      hex->food = 0;
      float fv = hex->food == 1 ? food_value : food_value * 2.0f;
      agent.health_points = min(max_hp, agent.health_points + fv);
      agent.score++;
    }
    agent.waiting += eating_waiting;
  }
};

class SpawningBehavior : public Behavior {
public:
  virtual void behave(Agent &agent, float perceptron_output) {
    int new_index = 0;
    while (new_index < num_agents && !agents[new_index].out) {
      new_index++;
    }
    if (new_index < num_agents) {
      int new_q = agent.q;
      int new_r = agent.r;
      int random_direction = agent.orientation;
      axial_add_direction(new_q, new_r, random_direction);
      WorldHex *hex = hex_axial(new_q, new_r);
      if (hex != 0 && hex->agent == 0) {
        agents[new_index].init_from_parent(&agent);
        agents[new_index].reset_agent(); 
        hex_axial(agents[new_index].q, agents[new_index].r)->agent = 0;
        agents[new_index].q = new_q;
        agents[new_index].r = new_r;
        agents[new_index].orientation = agent.orientation;
        hex_axial(agents[new_index].q, agents[new_index].r)->agent = &agents[new_index];
        agents[new_index].waiting += birth_waiting;
      }
    }
    agent.waiting += spawning_waiting;
  }
};

void init() {
  eg_init(WIDTH, HEIGHT, "Patterns of Life");
  setlocale(LC_NUMERIC, "");
}

Config cfg;
void refreshConfig() {
    try {
        cfg.readFile("config");
    } catch(const FileIOException &fioex) {
        printf("I/O error while reading file\n");  
        return;
    } catch(const ParseException &pex) {
        printf("Parse error\n");
        printf("File: %s\n", pex.getFile());
        printf("Line: %d\n", pex.getLine());
        printf("Error: %s\n", pex.getError());
        return;
    }

  Setting& root = cfg.getRoot();
  root.lookupValue("num_agents", num_agents);
  root.lookupValue("agent_spawn_rate", agent_spawn_rate);
  num_agents = min(num_agents, max_agents);
  root.lookupValue("food_spawn_rate", food_spawn_rate);
  root.lookupValue("food_value", food_value);
  root.lookupValue("max_hp", max_hp);
  root.lookupValue("rotational_waiting", rotational_waiting);
  root.lookupValue("linear_waiting", linear_waiting);
  root.lookupValue("eating_waiting", eating_waiting);
  root.lookupValue("kill_waiting", kill_waiting);
  root.lookupValue("spawning_waiting", spawning_waiting);
  root.lookupValue("birth_waiting", birth_waiting);
  root.lookupValue("burn_rate", burn_rate);
  root.lookupValue("turbo_rate", turbo_rate);
}

long last_refresh;
long last_refresh_interval = 1000000;

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
        frame_rate = int((float)turbo_rate * 0.04f);
        paused = false;
        break;
      case SDL_SCANCODE_3:
        frame_rate = int((float)turbo_rate * 0.20f);
        paused = false;
        break;
      case SDL_SCANCODE_4:
        frame_rate = turbo_rate;
        paused = false;
        break;
      case SDL_SCANCODE_5:
        frame_rate = turbo_rate * 5.0f;
        paused = false;
        break;
      case SDL_SCANCODE_6:
        frame_rate = turbo_rate * 5.0f * 5.0f;
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
          world[i].food = 0;
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
            remove_from_world(agents[num_agents - 1]);
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
          if (num_agents < max_agents) {
            agents[num_agents].randomize();
            agents[num_agents++].reset_agent();
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
  
  long now = system_clock::now().time_since_epoch().count();
  if (now - last_refresh > last_refresh_interval) {
    refreshConfig();
    last_refresh = now;
  }
  
  if (fdis(gen) < agent_spawn_rate) {
    for (int i = 0; i < num_agents; i++) {
      Agent &agent = agents[i];
      if (agent.out) {
        agent.randomize();
        agent.reset_agent();
        break;
      }
    }
  }

  // grow food
  if (fdis(gen) < food_spawn_rate) {
    world[(int)(fdis(gen) * WORLD_SIZE)].food |= 1;
  }

  // behavior model
  for (int i = 0; i < num_agents; i++) {

    Agent &agent = agents[i];

    if (agent.out) {
      continue;
    }

    // decay health as a function of time
    agent.health_points -= burn_rate;
    
    if (agent.waiting > 0) {
      agent.waiting--;
      continue;
    }

    // death
    if (agent.health_points <= 0.0f) {
      remove_from_world(agent);
      continue;
    }
    
    FoodSensor foodSensor_here(0, 0);
    FoodSensor foodSensor_ahead1(0, 1);
    FoodSensor foodSensor_ahead2(0, 2);
    FoodSensor foodSensor_ahead3(0, 3);
    FoodSensor foodSensor_left1(1, 1);
    FoodSensor foodSensor_left2(1, 2);
    FoodSensor foodSensor_right1(-1, 1);
    FoodSensor foodSensor_right2(-1, 2);
    AgentSensor agentSensor_ahead1(0, 1);
    SelfHealthPointsSensor selfHealthPointsSensor;
    
    float input1 = foodSensor_here.sense(agent);
    float input2 = foodSensor_ahead1.sense(agent);
    float input3 = foodSensor_ahead2.sense(agent);
    float input4 = foodSensor_ahead3.sense(agent); 
    float input5 = foodSensor_left1.sense(agent);
    float input6 = foodSensor_left2.sense(agent); 
    float input7 = foodSensor_right1.sense(agent);
    float input8 = foodSensor_right2.sense(agent); 
    float input9 = selfHealthPointsSensor.sense(agent); 
    float input10 = agent.memory[0];
    float input11 = agent.memory[1];
    float input12 = agent.memory[2];
    float input13 = agent.memory[3];
    float inputs[13] = { input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13 };
    
    float hidden[8] = { };
    float *weights = agent.dna;
    invoke_nn(13, inputs, 8, hidden, weights);
    weights += 13 * 8;
    float outputs[9] = { };
    invoke_nn(8, hidden, 9, outputs, weights);
    
    RotationalBehavior rotationalBehavior;
    LinearBehavior linearBehavior;
    KillBehavior killBehavior;
    EatingBehavior eatingBehavior;
    SpawningBehavior spawningBehavior;
    
    weights += (13 * 8 + 8 * 9);  
    
    if (outputs[0] > *weights++) {
      eatingBehavior.behave(agent, 1.0f);
    }
    
    if (outputs[1] > *weights++) {
      linearBehavior.behave(agent, 1.0f);
    }
    
    if (outputs[2] > *weights++) {
      killBehavior.behave(agent, 1.0f);
    }
    
    rotationalBehavior.behave(agent, outputs[3] * *weights++);
    
    if (outputs[4] > *weights++) {
      spawningBehavior.behave(agent, 1.0f);
    }
    
    agent.memory[0] = outputs[5] * *weights++;
    agent.memory[1] = outputs[6] * *weights++;
    agent.memory[2] = outputs[7] * *weights++;
    agent.memory[3] = outputs[8] * *weights++;
    
    assert(weights = agent.dna + DNA_SIZE);
  }
      
  // update record model
  if (frame % RECORD_SAMPLE_RATE == 0 && num_agents > 0) {
    int selected_index = select();
    records[records_index].selected_hue = agents[selected_index].hue;
    for (int i = 0; i < DNA_SIZE; i++) {
      records[records_index].dna[i] = agents[selected_index].dna[i];
    }
    for (int i = 0; i < max_agents; i++) {
      const Agent &agent = agents[i];
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
        axial_to_xy(agents[following].q, agents[following].r, x, y);
        camera_x = x;
        camera_y = y;
      }

      eg_scale(camera_zoom, camera_zoom);
      eg_translate(-camera_x, -camera_y);
      eg_translate((float)(WIDTH / 2) / camera_zoom,
                   (float)(HEIGHT / 2) / camera_zoom);

      for (int q = 0; q < Q; q++) {
        for (int r = 0; r < R; r++) {
          WorldHex *hex = hex_axial(q, r);

          eg_push_transform();
          int x, y;
          axial_to_xy(q, r, x, y);
          eg_translate(x, y);
          eg_scale(HEX_SIZE * 0.94F, HEX_SIZE * 0.94F);

          if (!(hex->food & 1))
            eg_set_color(0.1f, 0.2f, 0.05f, 1.0f);
          else
            eg_set_color(0.05f, 0.3f, 0.05f, 1.0f);
          
          glBegin(GL_POLYGON);
          glVertex2f(sin((M_PI * 1.5f) / 3.0f), cos((M_PI * 1.5f) / 3.0f));
          glVertex2f(sin((M_PI * 2.5f) / 3.0f), cos((M_PI * 2.5f) / 3.0f));
          glVertex2f(sin((M_PI * 3.5f) / 3.0f), cos((M_PI * 3.5f) / 3.0f));
          glVertex2f(sin((M_PI * 4.5f) / 3.0f), cos((M_PI * 4.5f) / 3.0f));
          glVertex2f(sin((M_PI * 5.5f) / 3.0f), cos((M_PI * 5.5f) / 3.0f));
          glVertex2f(sin((M_PI * 6.5f) / 3.0f), cos((M_PI * 6.5f) / 3.0f));
          glEnd();          
          
          if (hex->food & 2) { 
            eg_scale(.4f, .4f);
            eg_set_color(0.7f, 0.0f, 0.1f, 1.0f);
            glBegin(GL_POLYGON);
            glVertex2f(sin((2 * M_PI * 1.0f) / 3.0f), cos((2 * M_PI * 1.0f) / 3.0f));
            glVertex2f(sin((2 * M_PI * 2.0f) / 3.0f), cos((2 * M_PI * 2.0f) / 3.0f));
            glVertex2f(sin((2 * M_PI * 3.0f) / 3.0f), cos((2 * M_PI * 3.0f) / 3.0f));
            glEnd();
            eg_pop_transform();
          }
          
          eg_pop_transform();
        }
      }

      // draw agents
      for (int i = 0; i < num_agents; i++) {
        Agent agent = agents[i];
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
        float orientation_line_length = 25.0;
        eg_draw_line(x, 
                     y, 
                     x + (float)cos(angle) * orientation_line_length,
                     y + (float)sin(angle) * orientation_line_length,
                     15.0f);

        // agent
        eg_push_transform();
        eg_translate(x, y);
        eg_rotate((agent.orientation / 6.0f) * 360.0f + (360 / 12));
        const float AGENT_SIZE = 20.0f;
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
          if (agent.health_points > max_hp * 0.25f) {
            eg_set_color(0.5f, 0.9f, 0.5f, 0.8f);
          } else {
            eg_set_color(0.8f, 0.3f, 0.3f, 0.8f);
          }
          eg_draw_square(x - 15.0f, y + 12.0f, agent.health_points * 30.0f / max_hp, 5.0f);
        }
      }
    }

    // gene graph
    if (draw_record % 4 == 1) {
      float interval_h = HEIGHT / (float)DNA_SIZE;
      for (int rx = 0; rx < WIDTH; rx++) {
        const Record &record = records[(records_index + rx) % WIDTH];
        float mx = 0.0f;
        for (int wi = 0; wi < DNA_SIZE; wi++) {
          mx = fmax(mx, record.dna[wi]);
        }
        float y = 0.0f;
        for (int wi = 0; wi < DNA_SIZE; wi++) {
          float r, g, b;
          hsv_to_rgb(record.selected_hue, 1.0, 1.0, &r, &g, &b);
          eg_set_color(r, g, b, 1.0f);
          float h = (fabs(record.dna[wi]) / mx) * interval_h;
          eg_draw_line(rx, y, rx, y + h, 1.5f);
          y += interval_h;
        }
      }
    }

    // total score graph
    if (draw_record % 4 == 2) {
      for (int rx = 0; rx < WIDTH; rx++) {
        const Record &record = records[(records_index + rx) % WIDTH];
        for (int i = 0; i < max_agents; ++i) {
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
          hsv_to_rgb(record.hues[i], 1.00f, record.outs[i] ? 0.0f : 1.0f, &r, &g, &b);
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
  
}