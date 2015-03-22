#include "buddies.h"

struct Agent;

struct WorldHex {
  bool blocked;
  bool food;
  Agent *agent;
};

const int HEX_SIZE = 15;
const int WIDTH = 1280;
const int HEIGHT = 720;
const int Q = 114;
const int R = 80;
const int WORLD_SIZE = Q * R;

const int ANN_NUM_INPUT = 2;
const int ANN_NUM_HIDDEN = 10;
const int ANN_NUM_OUTPUT = 4;
const int ANN_NUM_CONNECTIONS = 74; // how to calculate this? ¯\_(ツ)_/¯

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

const int RECORD_SAMPLE_RATE = 100;
const int TURBO_RATE = 300; // how many simulation steps per render
const float AGENT_SIZE = 10.0f;
const float FOOD_VALUE = 100.0f;
const float EAT_DISTANCE = 16.0f;
const float MAX_HEALTH = 100.0f;

const int MAX_AGENTS = 200;

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> fdis(0, 1);

struct AgentInput {
  int forward_food_distance;
  float self_health;
};

struct AgentBehavior {
  int rotational; // rotational (twising and turning) behavior of the agent
  int linear; // forward moving (in the direction of orientation) behavior of
              // the agent
  bool eating;
  bool spawning;
};

struct Agent {
  int parent_index;
  float health;
  int score;
  fann *ann;
  int q, r, orientation;
  float hue;
  bool out;
};

struct Record {
  float weights[ANN_NUM_CONNECTIONS];
  int scores[MAX_AGENTS];
  float hues[MAX_AGENTS];
};

void deinit_agent(Agent *agent) {
  WorldHex *hex = hex_axial(agent->q, agent->r);
  if (hex != 0) {
    hex->agent = 0;
  }
}

void init_agent(Agent *agent) {

  deinit_agent(agent);

  fann_randomize_weights(agent->ann, -1.0f, 1.0f);
  fann_set_activation_function_hidden(agent->ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(agent->ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_steepness_hidden(agent->ann, 0.50f);
  fann_set_activation_steepness_output(agent->ann, 0.50f);
  // fann_set_training_algorithm(agent->ann, FANN_TRAIN_INCREMENTAL);
  // fann_set_learning_rate(agent->ann, 0.1f);

  agent->parent_index = -1;
  agent->health = MAX_HEALTH;
  agent->score = 0;
  agent->hue = fdis(gen);
  agent->out = false;
  WorldHex *hex;
  do {
    agent->q = Q * fdis(gen);
    agent->r = R * fdis(gen);
    hex = hex_axial(agent->q, agent->r);
  } while (hex == 0 || hex->agent != 0);
  agent->orientation = 6 * fdis(gen);
  hex_axial(agent->q, agent->r)->agent = agent;
}

static int frame = 0;

static Agent agents[MAX_AGENTS];

static int num_agents = 0;
static float food_spawn_rate = 0.001f;

static AgentBehavior agent_behaviors[MAX_AGENTS];

static Record records[WIDTH];
static int records_index = 0;
static int frame_rate = 1;
static bool pause = false;
static bool nudge = false;
static int draw_record = 0;
static bool draw_extra_info = false;
static bool delete_agent = false;
static int camera = 0;
static bool quit = false;

void unit_tests() {
  assert(0 == axial_distance(0, 0, 0, 0));
  assert(1 == axial_distance(0, 0, 1, 0));
  assert(2 == axial_distance(0, 0, 1, 1));
  assert(1 == axial_distance(0, 0, 0, 1));
  assert(10 == axial_distance(0, 0, 10, 0));

  fann *test_ann = fann_create_standard(3, ANN_NUM_INPUT, ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);
  printf("actual number of connections: %d (ANN_NUM_CONNECTIONS=%d)\n",
         fann_get_total_connections(test_ann), ANN_NUM_CONNECTIONS);
  assert(fann_get_total_connections(test_ann) == ANN_NUM_CONNECTIONS);
  fann_destroy(test_ann);
}

void init() {
  eg_init(WIDTH, HEIGHT, "Buddies3");
  for (int i = 0; i < MAX_AGENTS; i++) {
    agents[i].ann =
        fann_create_standard(3, ANN_NUM_INPUT, ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);
  }
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
        break;
      case SDL_SCANCODE_I:
        draw_extra_info = !draw_extra_info;
        break;
      case SDL_SCANCODE_D:
        delete_agent = true;
        break;
      case SDL_SCANCODE_Z:
        camera++;
        break;
      case SDL_SCANCODE_MINUS:
        if (e.keysym.mod & KMOD_SHIFT) {
          food_spawn_rate /= 2.0f;
        } else {
          if (num_agents > 0) {
            deinit_agent(&agents[num_agents - 1]);
            --num_agents;
          }
        }
        break;
      case SDL_SCANCODE_EQUALS:
        if (e.keysym.mod & KMOD_SHIFT) {
          food_spawn_rate *= 2.0f;
        } else {
          if (num_agents < MAX_AGENTS) {
            init_agent(&agents[num_agents++]);
          }
        }
        break;
      default:
        break;
      }
      break;
    }
    }
  }

  if (pause && !nudge)
    return;
  nudge = false;

  // respawn n % of out agents
  for (int i = 0; i < num_agents; i++) {
    if (agents[i].out && fdis(gen) < 0.01f) {
      init_agent(&agents[i]);
      agents[i].out = false;
    }
  }

  // spawn food
  if (fdis(gen) < food_spawn_rate) {
    world[(int)(fdis(gen) * WORLD_SIZE)].food = true;
  }

  // map world state to the agent input model
  AgentInput agent_inputs[num_agents];
  for (int i = 0; i < num_agents; i++) {
    if (agents[i].out) {
      continue;
    }
    int x, y, z;
    axial_to_cubic(agents[i].q, agents[i].r, x, y, z);
    for (int j = 0; j < 10; j++) {
      agent_inputs[i].forward_food_distance = j;
      WorldHex *hex = hex_cubic(x, y, z);
      if (hex != 0 && hex->food) {
        break;
      }
      cubic_add_direction(x, y, z, agents[i].orientation);
    }
    agent_inputs[i].self_health = agents[i].health / MAX_HEALTH;
  }

  // index of high scoring agent
  int high_score_index = 0;
  for (int i = 1; i < num_agents; i++) {
    if (agents[i].out) {
      continue;
    }
    if (agents[i].score > agents[high_score_index].score) {
      high_score_index = i;
    }
  }

  // behavior model
  for (int i = 0; i < num_agents; i++) {

    if (agents[i].out) {
      continue;
    }

    float ann_input[ANN_NUM_INPUT];
    ann_input[0] = agent_inputs[i].forward_food_distance;
    ann_input[1] = agent_inputs[i].self_health;

    // // train via mouse
    // int mouse_x, mouse_y;
    // SDL_PumpEvents();
    // if (SDL_GetMouseState(&mouse_x, &mouse_y) & SDL_BUTTON(SDL_BUTTON_LEFT))
    // {
    //   float dx_to_mouse = (float)mouse_x - agents[i].body->GetPosition().x;
    //   float dy_to_mouse =
    //       ((float)HEIGHT - (float)mouse_y) - agents[i].body->GetPosition().y;
    //   float fog_rotation = atan2(dy_to_mouse, dx_to_mouse);
    //   float delta_radians =
    //       angle_diff(fog_rotation, agents[i].body->GetAngle());
    //   float distance =
    //       sqrtf(dx_to_mouse * dx_to_mouse + dy_to_mouse * dy_to_mouse);
    //   float fog_rotational_behavior = max(min(delta_radians, +1.0f), -1.0f);
    //   float fog_linear_behavior = max(min(distance, +1.0f), -1.0f);
    //   float ann_output_train[ANN_NUM_OUTPUT] = {fog_rotational_behavior,
    //                                             fog_linear_behavior};
    //   fann_train(agents[i].ann, ann_input, ann_output_train);
    //   fog_flags[i] = true;
    //   fog_behaviors[i].rotational = fog_rotational_behavior;
    //   fog_behaviors[i].linear = fog_linear_behavior;
    // }

    // execute ann
    float *ann_output = fann_run(agents[i].ann, ann_input);
    agent_behaviors[i].rotational = (int)lroundf(ann_output[0] * 3.0f);
    agent_behaviors[i].linear = (int)lroundf(ann_output[1] * 3.0f);
    agent_behaviors[i].eating = ann_output[2] < 0.5f;
    agent_behaviors[i].spawning = ann_output[3] < 0.5f;

    // apply rotational behavior
    agents[i].orientation += agent_behaviors[i].rotational;
    while (agents[i].orientation < 0)
      agents[i].orientation += 6;
    while (agents[i].orientation > 5)
      agents[i].orientation -= 6;
    agents[i].health += (fabs(agent_behaviors[i].rotational) * -0.1f);

    // apply linear behavior
    hex_axial(agents[i].q, agents[i].r)->agent = 0;
    int x, y, z;
    axial_to_cubic(agents[i].q, agents[i].r, x, y, z);
    for (int j = 0; j < agent_behaviors[i].linear; j++) {
      int x0 = x, y0 = y, z0 = z;
      cubic_add_direction(x0, y0, z0, agents[i].orientation);
      WorldHex *hex = hex_cubic(x0, y0, z0);
      if (hex != 0 && hex->agent == 0) {
        x = x0;
        y = y0;
        z = z0;
        agents[i].health += -1.0f;
      }
    }
    cubic_to_axial(x, y, z, agents[i].q, agents[i].r);
    hex_axial(agents[i].q, agents[i].r)->agent = &agents[i];

    // decay health as a function of time
    agents[i].health += -0.1f;

    // handle d key flag
    if (delete_agent && i == high_score_index) {
      delete_agent = false;
      agents[i].health = -1.0f;
    }

    // eat near food
    if (agent_behaviors[i].eating) {
      WorldHex *hex = hex_axial(agents[i].q, agents[i].r);
      if (hex != 0 && hex->food) {
        hex->food = false;
        agents[i].health = min(MAX_HEALTH, agents[i].health + FOOD_VALUE);
        agents[i].score++;
      }
    }

    // death
    if (agents[i].health <= 0.0f) {
      hex_axial(agents[i].q, agents[i].r)->agent = 0;
      for (int j = 0; j < num_agents; j++) {
        if (agents[j].parent_index == i) {
          agents[j].parent_index = -1;
        }
      }
      agents[i].out = true;
      agents[i].score = 0;
    }

    // spawning
    if (!agents[i].out && agent_behaviors[i].spawning &&
        agents[i].health > MAX_HEALTH * 0.9f) {
      int new_index = 0;
      while (new_index < num_agents && !agents[new_index].out) {
        new_index++;
      }
      if (new_index != num_agents) {
        int new_q = agents[i].q;
        int new_r = agents[i].r;
        int random_direction = fdis(gen) * 6.0f;
        axial_add_direction(new_q, new_r, random_direction);
        WorldHex *hex = hex_axial(new_q, new_r);
        if (hex != 0 && hex->agent == 0) {
          init_agent(&agents[new_index]);
          hex_axial(agents[new_index].q, agents[new_index].r)->agent = 0;
          agents[new_index].out = false;
          agents[new_index].parent_index = i;
          agents[new_index].q = new_q;
          agents[new_index].r = new_r;
          hex_axial(agents[new_index].q, agents[new_index].r)->agent =
              &agents[new_index];
          fann *source_ann = agents[i].ann;
          fann *target_ann = agents[new_index].ann;
          int num_conn = fann_get_total_connections(source_ann);
          fann_connection source_connections[num_conn];
          fann_connection target_connections[num_conn];
          fann_get_connection_array(source_ann, source_connections);
          fann_get_connection_array(target_ann, target_connections);
          for (int j = 0; j < num_conn; j++) {
            if (fdis(gen) < 0.1f) {
              target_connections[j].weight =
                  source_connections[j].weight + ((fdis(gen) * 1.0f) - 0.5f);
            } else {
              target_connections[j].weight = source_connections[j].weight;
            }
          }
          fann_set_weight_array(target_ann, target_connections, num_conn);
          agents[new_index].hue = agents[i].hue;
          // share remaining health
          agents[new_index].health = agents[i].health = agents[i].health / 2.0f;
        }
      }
    }
  }

  // update the record model
  if (frame % RECORD_SAMPLE_RATE == 0 && num_agents > 0) {

    int selected_index = 0;

    int total_score = 0;
    for (int i = 0; i < num_agents; i++) {
      if (agents[i].out) {
        continue;
      }
      total_score += agents[i].score;
    }

    int random_score = (int)(fdis(gen) * (float)total_score);
    for (int i = 0; i < num_agents && random_score >= 0.0f; i++) {
      if (agents[i].out) {
        continue;
      }
      random_score -= agents[i].score;
      selected_index = i;
    }

    fann *ann = agents[selected_index].ann;
    int num_conn = fann_get_total_connections(ann);
    fann_connection connections[num_conn];
    fann_get_connection_array(ann, connections);
    for (int i = 0; i < num_conn; i++) {
      records[records_index].weights[i] = connections[i].weight;
    }
    for (int i = 0; i < num_agents; i++) {
      records[records_index].scores[i] = agents[i].score;
      records[records_index].hues[i] = agents[i].hue;
    }
    records_index++;
    records_index %= WIDTH;
  }

  if (frame % frame_rate == 0) {

    eg_clear_screen(0.0f, 0.0f, 0.0f, 0.0f);
    eg_reset_transform();

    static float hex_x0 = cos(0 * (M_PI / 3));
    static float hex_y0 = sin(0 * (M_PI / 3));
    static float hex_x1 = cos(1 * (M_PI / 3));
    static float hex_y1 = sin(1 * (M_PI / 3));
    static float hex_x2 = cos(2 * (M_PI / 3));
    static float hex_y2 = sin(2 * (M_PI / 3));
    static float hex_x3 = cos(3 * (M_PI / 3));
    static float hex_y3 = sin(3 * (M_PI / 3));
    static float hex_x4 = cos(4 * (M_PI / 3));
    static float hex_y4 = sin(4 * (M_PI / 3));
    static float hex_x5 = cos(5 * (M_PI / 3));
    static float hex_y5 = sin(5 * (M_PI / 3));

    if (draw_record % 3 == 0) {

      if (camera % 4 == 0) {
        eg_translate(0, -HEX_SIZE * R * 0.5f);
        eg_scale(0.5f, 0.5f);
      } else if (camera % 4 == 1) {
        eg_translate(0, -HEX_SIZE * R * 1.0f);
        eg_scale(1.0f, 1.0f);
      } else if (camera % 4 == 2) {
        eg_translate(0, -HEX_SIZE * R * 2.0f);
        eg_scale(2.0f, 2.0f);
      } else if (camera % 4 == 3) {
        eg_translate(0, -HEX_SIZE * R * 0.25f);
        eg_scale(0.25f, 0.25f);
      }
      for (int q = 0; q < Q; q++) {
        for (int r = 0; r < R; r++) {
          eg_push_transform();
          int x, y;
          axial_to_xy(q, r, x, y);
          eg_translate(x, y);
          eg_scale(HEX_SIZE, HEX_SIZE);
          eg_set_color(0.3f, 0.3f, 0.3f, 1.0f);
          glLineWidth(1);
          glBegin(GL_LINE_LOOP);
          glVertex2f(hex_x0, hex_y0);
          glVertex2f(hex_x1, hex_y1);
          glVertex2f(hex_x2, hex_y2);
          glVertex2f(hex_x3, hex_y3);
          glVertex2f(hex_x4, hex_y4);
          glVertex2f(hex_x5, hex_y5);
          glEnd();

          // draw foods
          if (hex_axial(q, r)->food) {
            eg_scale(0.5f, 0.5f);
            eg_set_color(0.1f, 0.9f, 0.1f, 1.0f);
            eg_draw_square(-0.5f, -0.5, 1, 1);
          }

          eg_pop_transform();
        }
      }

      for (int i = 0; i < num_agents; i++) {
        if (agents[i].out) {
          continue;
        }

        // pixel location
        int x, y;
        axial_to_xy(agents[i].q, agents[i].r, x, y);

        // indicate orientation
        float r, g, b;
        hsv_to_rgb(agents[i].hue, 1.0f, 1.0f, &r, &g, &b);
        eg_set_color(r, g, b, 1.0f);
        float angle = agents[i].orientation / 6.0f * 2 * M_PI + (M_PI / 6.0f);
        float orientation_line_length =
            agent_behaviors[i].eating ? 20.0f : 10.0f;
        eg_draw_line(x, y, x + (float)cos(angle) * orientation_line_length,
                     y + (float)sin(angle) * orientation_line_length,
                     agents[i].parent_index == -1 ? 10.0f : 5.0f);

        // agent
        float buddy_size =
            AGENT_SIZE * (agents[i].parent_index == -1 ? 1.0f : 0.6f);
        eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
        eg_draw_square(x - 0.5f * buddy_size, y - 0.5f * buddy_size, buddy_size,
                       buddy_size);
        eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
        eg_draw_square(x - 0.5f * buddy_size * 0.5f,
                       y - 0.5f * buddy_size * 0.5f, buddy_size * 0.5f,
                       buddy_size * 0.5f);

        if (draw_extra_info) {
          // health bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(x - 15.0f, y + 12.0f, 30.0f, 5.0f);
          if (agents[i].health > MAX_HEALTH * 0.25f) {
            eg_set_color(0.5f, 0.9f, 0.5f, 0.8f);
          } else {
            eg_set_color(0.8f, 0.3f, 0.3f, 0.8f);
          }
          eg_draw_square(x - 15.0f, y + 12.0f,
                         agents[i].health * 30.0f / MAX_HEALTH, 5.0f);

          // score bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(x - 15.0f, y + 20.0f, 30.0f, 5.0f);
          eg_set_color(0.9f, 0.85f, 0.0f, 0.8f);
          eg_draw_square(x - 15.0f, y + 20.0f,
                         30.0f * ((float)agents[i].score /
                                  (float)agents[high_score_index].score),
                         5.0f);
        }
        if (i == high_score_index) {
          eg_set_color(0.9f, 0.3f, 0.3f, 1.0f);
          eg_draw_square(x - 0.5f * buddy_size, y - 0.5f * buddy_size,
                         buddy_size, buddy_size);
        }
      }
    }

    // ann weight graph
    if (draw_record % 3 == 1) {
      for (int rx = 0; rx < WIDTH; rx++) {
        Record record = records[(records_index + rx) % WIDTH];
        float y = 0.0f;
        float total_weight = 0.0f;
        for (int wi = 0; wi < ANN_NUM_CONNECTIONS; wi++) {
          total_weight += fabs(record.weights[wi]);
        }
        for (int wi = 0; wi < ANN_NUM_CONNECTIONS; wi++) {
          float r, g, b;
          float hue = (float)wi / (float)ANN_NUM_CONNECTIONS * 7.0f;
          while (hue > 1.0f)
            hue -= 1.0f;
          hsv_to_rgb(hue, 0.90f, 1.00f, &r, &g, &b);
          eg_set_color(r, g, b, 1.0f);
          float h = (fabs(record.weights[wi]) / total_weight) * HEIGHT * 2;
          eg_draw_line(rx, y, rx, y + h, 1.5f);
          y += h;
        }
      }
    }

    // total score graph
    if (draw_record % 3 == 2) {
      for (int rx = 0; rx < WIDTH; rx++) {
        Record record = records[(records_index + rx) % WIDTH];
        for (int i = 0; i < num_agents; ++i) {
          float r, g, b;
          hsv_to_rgb(record.hues[i], 1.00f, 1.00f, &r, &g, &b);
          eg_set_color(r, g, b, 0.8f);
          eg_draw_square(rx, record.scores[i] % HEIGHT, 1.0f, 1.0f);
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

  printf("%d\n", frame);

  eg_shutdown();

  return 0;
}
