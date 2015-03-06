#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <set>
#include <fann.h>
#include <Box2D/Box2D.h>
#ifdef __EMSCRIPTEN__
#include "easygame_emscripten.h"
#include <emscripten.h>
#else
#include "easygame.h"
#endif

using std::min;
using std::max;

const int RECORD_SAMPLE_RATE = 100;
const int TURBO_RATE = 300; // how many simulation steps per render
const int WIDTH = 1280;
const int HEIGHT = 720;
const int GRID_WIDTH = WIDTH / 20;
const int GRID_HEIGHT = HEIGHT / 20;
const float GRID_CELL_WIDTH = (float)WIDTH / (float)GRID_WIDTH;
const float GRID_CELL_HEIGHT = (float)HEIGHT / (float)GRID_HEIGHT;
const float BUDDY_SIZE = 10.0f;
const float FOOD_SIZE = 6.0f;
const float FOOD_VALUE = 100.0f;
const float EAT_DISTANCE = 16.0f;
const float MAX_HEALTH = 100.0f;

const int MAX_AGENTS = 100;
const int MAX_FOODS = 1000;

const int ANN_NUM_INPUT = 3;
const int ANN_NUM_HIDDEN = 8;
const int ANN_NUM_OUTPUT = 4;
const int ANN_NUM_CONNECTIONS = 68; // how to calculate this? ¯\_(ツ)_/¯

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> fdis(0, 1);

enum GridCell { GridCellEmpty, GridCellFull };

struct Grid {
  GridCell cells[GRID_WIDTH * GRID_HEIGHT];

  GridCell &cell(int x, int y) {
    assert(x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT);
    return cells[y * GRID_WIDTH + x];
  }

  GridCell &cell_at(float x, float y) {
    assert(x >= 0.0f && x <= WIDTH && y >= 0.0f && y <= HEIGHT);
    int ix = clamp((int)(x / GRID_CELL_WIDTH), 0, GRID_WIDTH - 1);
    int iy = clamp((int)(y / GRID_CELL_HEIGHT), 0, GRID_HEIGHT - 1);
    return cell(ix, iy);
  }
};

struct Food {
  b2Body *body;
};

struct AgentInput {
  float nearest_food_relative_direction;
  float nearest_food_distance;
  float self_health;
};

struct AgentBehavior {
  float rotational; // rotational (twising and turning) behavior of the agent
  float linear; // forward moving (in the direction of orientation) behavior of
                // the agent
  bool eating;
  bool spawning;
};

struct Agent {
  int parent_index;
  float health;
  int score;
  fann *ann;
  b2Body *body;
  float hue;
  bool out;
};

struct Record {
  float weights[ANN_NUM_CONNECTIONS];
  int scores[MAX_AGENTS];
  float hues[MAX_AGENTS];
};

b2Vec2 gravity(0.0f, 0.0f);
b2World world(gravity);

void deinit_agent(Agent *agent) {
  if (agent->ann) {
    fann_destroy(agent->ann);
    agent->ann = 0;
  }
  if (agent->body) {
    world.DestroyBody(agent->body);
    agent->body = 0;
  }
}

void init_agent(Agent *agent) {

  deinit_agent(agent);

  b2BodyDef bodyDef;
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(WIDTH * fdis(gen), HEIGHT * fdis(gen));
  bodyDef.angle = fdis(gen) * 2 * M_PI - M_PI;
  bodyDef.linearDamping = .50f;
  bodyDef.angularDamping = 5.0f;
  bodyDef.gravityScale = 1.0f;

  b2PolygonShape dynamicBox;
  dynamicBox.SetAsBox(BUDDY_SIZE / 2.0f, BUDDY_SIZE / 2.0f);

  b2FixtureDef fixtureDef;
  fixtureDef.shape = &dynamicBox;
  fixtureDef.friction = 10.0f;
  fixtureDef.density = .01f;

  b2Body *b = world.CreateBody(&bodyDef);
  b->CreateFixture(&fixtureDef);

  agent->ann =
      fann_create_standard(3, ANN_NUM_INPUT, ANN_NUM_HIDDEN, ANN_NUM_OUTPUT);

  fann_randomize_weights(agent->ann, -1.0f, 1.0f);
  fann_set_activation_function_hidden(agent->ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(agent->ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_steepness_hidden(agent->ann, 0.10f);
  fann_set_activation_steepness_output(agent->ann, 0.19f);
  fann_set_training_algorithm(agent->ann, FANN_TRAIN_INCREMENTAL);
  fann_set_learning_rate(agent->ann, 0.1f);

  agent->parent_index = -1;
  agent->health = MAX_HEALTH;
  agent->score = 0;
  agent->body = b;
  agent->hue = fdis(gen);
  agent->out = false;
}

void deinit_food(Food *food) {
  if (food->body) {
    world.DestroyBody(food->body);
    food->body = 0;
  }
}

void init_food(Food *food) {

  deinit_food(food);

  b2BodyDef bodyDef;
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(WIDTH * fdis(gen), HEIGHT * fdis(gen));
  bodyDef.angle = 0.0f;
  bodyDef.linearDamping = 0.0f;
  bodyDef.angularDamping = 0.0f;
  bodyDef.gravityScale = 1.0f;

  b2Body *b = world.CreateBody(&bodyDef);

  b2PolygonShape dynamicBox;
  dynamicBox.SetAsBox(FOOD_SIZE / 2, FOOD_SIZE / 2);

  b2FixtureDef fixtureDef;
  fixtureDef.shape = &dynamicBox;
  fixtureDef.friction = 1.0f;
  fixtureDef.density = 0.001f;
  fixtureDef.restitution = 0.5f;

  b->CreateFixture(&fixtureDef);

  food->body = b;
}

static int frame = 0;
static Grid grid;
static Agent agents[MAX_AGENTS];
static int num_agents = 1;
static AgentBehavior agent_behaviors[MAX_AGENTS];
static AgentBehavior fog_behaviors[MAX_AGENTS];
static bool fog_flags[MAX_AGENTS];
static Food foods[MAX_FOODS];
static int num_foods = 1;
static bool quit = false;
static EGSound *pickup_sound;
static Record records[WIDTH];
static int records_index = 0;
static int frame_rate = 1;
static bool pause = false;
static bool nudge = false;
static int draw_record = 0;
static bool draw_extra_info = false;
static bool delete_agent = false;
static int camera = 0;

void init() {
  eg_init(WIDTH, HEIGHT, "Buddies");
  pickup_sound = eg_load_sound("assets/pickup.wav");
  for (int y = 0; y < GRID_HEIGHT; y++) {
    for (int x = 0; x < GRID_WIDTH; x++) {
      grid.cell(x, y) = GridCellEmpty;
    }
  }
  init_agent(&agents[0]);
  init_food(&foods[0]);
  printf("actual number of connections: %d (ANN_NUM_CONNECTIONS=%d)\n",
         fann_get_total_connections(agents[0].ann), ANN_NUM_CONNECTIONS);
  assert(fann_get_total_connections(agents[0].ann) == ANN_NUM_CONNECTIONS);
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
      // printf("SDL_MOUSEBUTTONDOWN button=%d state=%d x=%d y=%d\n", e.button,
      //  e.state, e.x, e.y);
      break;
    }
    case SDL_KEYDOWN: {
      SDL_KeyboardEvent e = event.key;
      // printf("SDL_KEYDOWN scancode=%d\n", e.keysym.scancode);
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
          for (int i = 0; i < 10 && num_foods != 1; i++) {
            deinit_food(&foods[num_foods - 1]);
            --num_foods;
          }
        } else {
          if (num_agents != 1) {
            deinit_agent(&agents[num_agents - 1]);
            --num_agents;
          }
        }
        break;
      case SDL_SCANCODE_EQUALS:
        if (e.keysym.mod & KMOD_SHIFT) {
          for (int i = 0; i < 10 && num_foods < MAX_FOODS; i++) {
            init_food(&foods[num_foods++]);
          }
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

  for (int i = 0; i < num_foods; i++) {
    if (foods[i].body->GetPosition().x < 0 ||
        foods[i].body->GetPosition().x > WIDTH ||
        foods[i].body->GetPosition().y < 0 ||
        foods[i].body->GetPosition().y > HEIGHT) {
      init_food(&foods[i]);
    }
  }

  float32 timeStep = 1.0f / 60.0f;
  int32 velocityIterations = 6;
  int32 positionIterations = 2;
  world.Step(timeStep, velocityIterations, positionIterations);

  // respawn n % of out agents
  for (int i = 0; i < num_agents; i++) {
    if (agents[i].out && fdis(gen) < 0.001f) {
      init_agent(&agents[i]);
      agents[i].out = false;
    }
  }

  // food index maps every agent to its nearest food
  int food_index[num_agents];
  for (int i = 0; i < num_agents; i++) {
    // if (agents[i].out) {
    //   continue;
    // }
    float nearest_dist_sq;
    for (int j = 0; j < num_foods; j++) {
      float dx =
          foods[j].body->GetPosition().x - agents[i].body->GetPosition().x;
      float dy =
          foods[j].body->GetPosition().y - agents[i].body->GetPosition().y;
      float dist_sq = (dx * dx) + (dy * dy);
      if (j == 0 || dist_sq < nearest_dist_sq) {
        food_index[i] = j;
        nearest_dist_sq = dist_sq;
      }
    }
  }

  // map world state to the agent input model
  AgentInput agent_inputs[num_agents];
  for (int i = 0; i < num_agents; i++) {
    if (agents[i].out) {
      continue;
    }
    float dx = foods[food_index[i]].body->GetPosition().x -
               agents[i].body->GetPosition().x;
    float dy = foods[food_index[i]].body->GetPosition().y -
               agents[i].body->GetPosition().y;
    agent_inputs[i].nearest_food_relative_direction =
        angle_diff(atan2(dy, dx), agents[i].body->GetAngle());
    agent_inputs[i].nearest_food_distance = sqrtf(dx * dx + dy * dy);
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
    ann_input[0] = agent_inputs[i].nearest_food_relative_direction;
    ann_input[1] = agent_inputs[i].nearest_food_distance;
    ann_input[2] = agent_inputs[i].self_health;

    fog_flags[i] = false;

    // train via mouse
    int mouse_x, mouse_y;
    SDL_PumpEvents();
    if (SDL_GetMouseState(&mouse_x, &mouse_y) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      float dx_to_mouse = (float)mouse_x - agents[i].body->GetPosition().x;
      float dy_to_mouse =
          ((float)HEIGHT - (float)mouse_y) - agents[i].body->GetPosition().y;
      float fog_rotation = atan2(dy_to_mouse, dx_to_mouse);
      float delta_radians =
          angle_diff(fog_rotation, agents[i].body->GetAngle());
      float distance =
          sqrtf(dx_to_mouse * dx_to_mouse + dy_to_mouse * dy_to_mouse);
      float fog_rotational_behavior = max(min(delta_radians, +1.0f), -1.0f);
      float fog_linear_behavior = max(min(distance, +1.0f), -1.0f);
      float ann_output_train[ANN_NUM_OUTPUT] = {fog_rotational_behavior,
                                                fog_linear_behavior};
      fann_train(agents[i].ann, ann_input, ann_output_train);
      fog_flags[i] = true;
      fog_behaviors[i].rotational = fog_rotational_behavior;
      fog_behaviors[i].linear = fog_linear_behavior;
    }

    // execute ann
    float *ann_output = fann_run(agents[i].ann, ann_input);
    agent_behaviors[i].rotational = ann_output[0];
    agent_behaviors[i].linear = ann_output[1];
    agent_behaviors[i].eating = ann_output[2] < 0.5f;
    agent_behaviors[i].spawning = ann_output[3] < 0.5f;

    // apply behavior to box 2d model
    agents[i].body->ApplyLinearImpulse(
        b2Vec2(agent_behaviors[i].linear * cos(agents[i].body->GetAngle()),
               agent_behaviors[i].linear * sin(agents[i].body->GetAngle())),
        agents[i].body->GetPosition(), true);
    agents[i].body->ApplyAngularImpulse(agent_behaviors[i].rotational, true);

    // decay health as a function of time
    agents[i].health += -0.01f;

    // handle d key flag
    if (delete_agent && i == high_score_index) {
      delete_agent = false;
      agents[i].health = -1.0f;
    }

    // eat near food
    if (agent_behaviors[i].eating) {
      for (int j = 0; j < num_foods; j++) {
        float dx =
            foods[j].body->GetPosition().x - agents[i].body->GetPosition().x;
        float dy =
            foods[j].body->GetPosition().y - agents[i].body->GetPosition().y;
        float dist_sq = (dx * dx) + (dy * dy);
        if (dist_sq < EAT_DISTANCE * EAT_DISTANCE) {
          agents[i].health = min(MAX_HEALTH, agents[i].health + FOOD_VALUE);
          agents[i].score++;
          init_food(&foods[j]);
          break;
        }
      }
    }

    // death
    if (agents[i].health <= 0.0f) {

      for (int j = 0; j < num_agents; j++) {
        if (agents[j].parent_index == i) {
          agents[j].parent_index = -1;
        }
      }

      agents[i].body->SetActive(false);
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
        init_agent(&agents[new_index]);
        agents[new_index].body->SetActive(true);
        agents[new_index].out = false;
        agents[new_index].parent_index = i;
        agents[new_index].body->SetTransform(
            agents[i].body->GetPosition(), agents[new_index].body->GetAngle());
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

  // update the record model
  if (frame % RECORD_SAMPLE_RATE == 0) {

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

    records_index++;
    records_index %= WIDTH;
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
  }

  bool skip_render = (frame % frame_rate != 0);
  if (!skip_render) {

    eg_reset_transform();
    if (camera % 3 == 0) {
      eg_scale(0.5f, 0.5f);
      eg_translate(WIDTH / 2, HEIGHT / 2);
    } else if (camera % 3 == 1) {
      eg_scale(2.0f, 2.0f);
      eg_translate(
          -agents[high_score_index].body->GetPosition().x + (WIDTH / 4),
          -agents[high_score_index].body->GetPosition().y + (HEIGHT / 4));
    }
    eg_clear_screen(0.0f, 0.0f, 0.0f, 0.0f);

    // draw agents
    if (draw_record % 3 == 0) {
      // draw grid
      eg_set_color(0.5f, 0.5f, 0.5f, 0.25f);
      for (int x = 0; x < GRID_WIDTH; x++) {
        float fx = x / (float)GRID_WIDTH * WIDTH;
        eg_draw_line(fx, 0.0f, fx, HEIGHT);
        if (x % 7 == 0) {
          eg_set_color(1.0f, 1.0f, 1.0f, 0.15f);
          eg_draw_line(fx, 0.0f, fx, HEIGHT);
          eg_set_color(0.5f, 0.5f, 0.5f, 0.15f);
        }
      }
      for (int y = 0; y < GRID_HEIGHT; y++) {
        float fy = y / (float)GRID_WIDTH * WIDTH;
        eg_draw_line(0.0f, fy, WIDTH, fy);
        if (y % 7 == 0) {
          eg_set_color(1.0f, 1.0f, 1.0f, 0.15f);
          eg_draw_line(0.0f, fy, WIDTH, fy);
          eg_set_color(0.5f, 0.5f, 0.5f, 0.15f);
        }
      }

      for (int i = 0; i < num_agents; i++) {
        if (agents[i].out) {
          continue;
        }

        // indicate orientation
        float r, g, b;
        hsv_to_rgb(agents[i].hue, 1.0f, 1.0f, &r, &g, &b);
        eg_set_color(r, g, b, 1.0f);
        float orientation_line_length =
            agent_behaviors[i].eating ? 20.0f : 10.0f;
        eg_draw_line(agents[i].body->GetPosition().x,
                     agents[i].body->GetPosition().y,
                     agents[i].body->GetPosition().x +
                         (float)cos(agents[i].body->GetAngle()) *
                             orientation_line_length,
                     agents[i].body->GetPosition().y +
                         (float)sin(agents[i].body->GetAngle()) *
                             orientation_line_length,
                     agents[i].parent_index == -1 ? 10.0f : 5.0f);

        // indicate nearest food (by food index)
        if (draw_extra_info) {
          eg_set_color(0.9f, 0.9f, 0.9f, 0.3f);
          eg_draw_line(agents[i].body->GetPosition().x,
                       agents[i].body->GetPosition().y,
                       foods[food_index[i]].body->GetPosition().x,
                       foods[food_index[i]].body->GetPosition().y, 2);
        }

        // agent
        float buddy_size =
            BUDDY_SIZE * (agents[i].parent_index == -1 ? 1.0f : 0.6f);
        eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
        eg_draw_square(agents[i].body->GetPosition().x - 0.5f * buddy_size,
                       agents[i].body->GetPosition().y - 0.5f * buddy_size,
                       buddy_size, buddy_size);
        eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
        eg_draw_square(
            agents[i].body->GetPosition().x - 0.5f * buddy_size * 0.5f,
            agents[i].body->GetPosition().y - 0.5f * buddy_size * 0.5f,
            buddy_size * 0.5f, buddy_size * 0.5f);

        if (draw_extra_info) {
          // health bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(agents[i].body->GetPosition().x - 15.0f,
                         agents[i].body->GetPosition().y + 12.0f, 30.0f, 5.0f);
          if (agents[i].health > MAX_HEALTH * 0.25f) {
            eg_set_color(0.5f, 0.9f, 0.5f, 0.8f);
          } else {
            eg_set_color(0.8f, 0.3f, 0.3f, 0.8f);
          }
          eg_draw_square(agents[i].body->GetPosition().x - 15.0f,
                         agents[i].body->GetPosition().y + 12.0f,
                         agents[i].health * 30.0f / MAX_HEALTH, 5.0f);

          // score bar
          eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
          eg_draw_square(agents[i].body->GetPosition().x - 15.0f,
                         agents[i].body->GetPosition().y + 20.0f, 30.0f, 5.0f);
          eg_set_color(0.9f, 0.85f, 0.0f, 0.8f);
          eg_draw_square(agents[i].body->GetPosition().x - 15.0f,
                         agents[i].body->GetPosition().y + 20.0f,
                         30.0f * ((float)agents[i].score /
                                  (float)agents[high_score_index].score),
                         5.0f);
        }

        // high score
        if (i == high_score_index) {
          eg_set_color(0.9f, 0.3f, 0.3f, 1.0f);
          eg_draw_square(agents[high_score_index].body->GetPosition().x -
                             0.5f * BUDDY_SIZE,
                         agents[high_score_index].body->GetPosition().y -
                             0.5f * BUDDY_SIZE,
                         BUDDY_SIZE, BUDDY_SIZE);
        }
      }

      // draw foods
      for (int i = 0; i < num_foods; i++) {
        eg_push_transform();
        eg_set_color(0.0f, 0.8f, 0.0f, 1.0f);
        eg_translate(foods[i].body->GetPosition().x,
                     foods[i].body->GetPosition().y);
        eg_rotate(foods[i].body->GetAngle());
        eg_draw_square(-0.5f * FOOD_SIZE, -0.5 * FOOD_SIZE, FOOD_SIZE,
                       FOOD_SIZE);
        eg_pop_transform();
      }
    }

    // record model
    if (draw_record % 3 != 0) {
      eg_reset_transform();

      for (int rx = 0; rx < WIDTH; rx++) {
        Record record = records[(records_index + WIDTH - rx) % WIDTH];

        // ann weight graph
        if (draw_record % 3 == 1) {
          float y = 0.0f;
          float total_weight = 0.0f;
          for (int wi = 0; wi < ANN_NUM_CONNECTIONS; wi++) {
            total_weight += fabs(record.weights[wi]);
          }
          for (int wi = 0; wi < ANN_NUM_CONNECTIONS; wi++) {
            float r, g, b;
            float hue = (float)wi / (float)ANN_NUM_CONNECTIONS * 20.0f;
            while (hue > 1.0f)
              hue -= 1.0f;
            hsv_to_rgb(hue, 0.90f, 1.00f, &r, &g, &b);
            eg_set_color(r, g, b, 1.0f);
            float h = (fabs(record.weights[wi]) / total_weight) * HEIGHT * 2;
            eg_draw_line(rx, y, rx, y + h, 1.5f);
            y += h;
          }
        }

        // total score graph
        if (draw_record % 3 == 2) {
          for (int i = 0; i < num_agents; ++i) {
            float r, g, b;
            hsv_to_rgb(record.hues[i], 1.00f, 1.00f, &r, &g, &b);
            eg_set_color(r, g, b, 0.8f);
            eg_draw_square(rx, HEIGHT - (record.scores[i] % HEIGHT), 1.0f,
                           1.0f);
          }
        }
      }
    }

    eg_swap_buffers();
  }

  frame++;
}

int main(int argc, char *argv[]) {

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
