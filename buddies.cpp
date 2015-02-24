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

const int   TURBO_RATE = 240; // how many simulation steps per render
const int   WIDTH = 1280;
const int   HEIGHT = 720;
const int   GRID_WIDTH = WIDTH / 20;
const int   GRID_HEIGHT = HEIGHT / 20;
const float GRID_CELL_WIDTH = (float)WIDTH / (float)GRID_WIDTH;
const float GRID_CELL_HEIGHT = (float)HEIGHT / (float)GRID_HEIGHT;
const float BUDDY_SIZE = 10.0f;
const float FOOD_SIZE = 6.0f;
const int   FOOD_COUNT = 4;
const float FOOD_VALUE =  100.0f;
const float EAT_DISTANCE = 16.0f;
const float HEALTH_DECAY_CONSTANT = 0.01f;
const float MAX_HEALTH = 100.0f;
const int   RECORD_SAMPLE_RATE = 1;

const int   NUM_AGENTS = 1;

const int   ANN_NUM_INPUT = 2;
const int   ANN_NUM_HIDDEN = 10;
const int   ANN_NUM_OUTPUT = 2;
const int   ANN_NUM_CONNECTIONS = 6; // how to calculate this? ¯\_(ツ)_/¯

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
  b2Body *body;
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
  float health;
  int score;
  fann *ann;
  b2Body *body;
  float hue;
};

struct Record {
  float weights[ANN_NUM_CONNECTIONS];
    int scores[NUM_AGENTS];
  float mse;
  float rotational;
  float linear;
   bool fog_on;
  float fog_rotational;
  float fog_linear;
};

b2Vec2 gravity(0.0f, 0.0f);
b2World world(gravity);

void init_agent(Agent *agent) {

  if (agent->body) {
    world.DestroyBody(agent->body);
  }

  b2BodyDef bodyDef;
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(WIDTH * fdis(gen), HEIGHT * fdis(gen));
  bodyDef.angle = fdis(gen) * 2 * M_PI - M_PI;
  bodyDef.linearDamping = .50f;
  bodyDef.angularDamping = 5.0f;
  bodyDef.gravityScale = 0.0f;

  b2PolygonShape dynamicBox;
  dynamicBox.SetAsBox(BUDDY_SIZE / 2.0f, BUDDY_SIZE / 2.0f);

  b2FixtureDef fixtureDef;
  fixtureDef.shape = &dynamicBox;
  fixtureDef.friction = 1.0f;
  fixtureDef.density = .01f;

  b2Body *b = world.CreateBody(&bodyDef);
  b->CreateFixture(&fixtureDef);

  if (agent->ann) {
    fann_destroy(agent->ann);
  }

  agent->ann = fann_create_standard(2, ANN_NUM_INPUT, ANN_NUM_OUTPUT);

                 fann_randomize_weights(agent->ann, -0.1f, +0.1f);
    fann_set_activation_function_hidden(agent->ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(agent->ann, FANN_SIGMOID_SYMMETRIC);
   fann_set_activation_steepness_hidden(agent->ann, 0.02f);
   fann_set_activation_steepness_output(agent->ann, 0.19f);
            fann_set_training_algorithm(agent->ann, FANN_TRAIN_INCREMENTAL);
                 fann_set_learning_rate(agent->ann, 0.01f);

  agent->parent_index = -1;
  agent->health = MAX_HEALTH;
  agent->score = 0;
  agent->body = b;
  agent->hue = fdis(gen);
}

void init_food(Food *food) {
  if (food->body) {
    world.DestroyBody(food->body);
  }

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
  fixtureDef.density = 1.0f;
  fixtureDef.restitution = 0.5f;

  b->CreateFixture(&fixtureDef);

  food->body = b;
}

void print_ann(fann *ann) {
  fann_print_connections(ann);
  printf("MSE: %f\n", fann_get_MSE(ann));
}

static        int frame               = 0     ;
static       Grid grid                        ;
static      Agent agents[NUM_AGENTS]          ;
static AgentBehavior agent_behaviors[NUM_AGENTS] ;
static AgentBehavior fog_behaviors[NUM_AGENTS]   ;
static bool          fog_flags[NUM_AGENTS]       ;
static       Food foods[FOOD_COUNT]           ;
static       bool quit                = false ;
static    EGSound *pickup_sound               ;
static     Record records[WIDTH]              ;
static        int records_index       = 0     ;
static        int frame_rate          = 1     ;
static       bool pause               = false ;
static       bool nudge               = false ;
static       bool draw_record         = false ;
static       bool delete_agent        = false ;

void init() {
  //
  // b2BodyDef groundBodyDef;
  // groundBodyDef.position.Set(WIDTH / 2.0f, 0.0f);
  // groundBodyDef.angle = 0.0f;
  // b2Body* groundBody = world.CreateBody(&groundBodyDef);
  // b2PolygonShape groundBox;
  // groundBox.SetAsBox(WIDTH / 2.0f, 1.0f);
  // groundBody->CreateFixture(&groundBox, 0.0f);

  eg_init(WIDTH, HEIGHT, "Buddies");

  pickup_sound = eg_load_sound("assets/pickup.wav");

  for(int y = 0; y < GRID_HEIGHT; y++) {
    for(int x = 0; x < GRID_WIDTH; x++) {
      grid.cell(x, y) = GridCellEmpty;
    }
  }

  for(int i = 0; i < NUM_AGENTS; i++) {
    init_agent(&agents[i]);
  }

  for (int i = 0; i < FOOD_COUNT; i++) {
    init_food(&foods[i]);
  }

  printf("actual number of connections: %d (ANN_NUM_CONNECTIONS=%d)\n", fann_get_total_connections(agents[0].ann), ANN_NUM_CONNECTIONS);
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
        printf("SDL_MOUSEBUTTONDOWN button=%d state=%d x=%d y=%d\n", e.button, e.state, e.x, e.y);
        break;
      }
      case SDL_KEYDOWN: {
        SDL_KeyboardEvent e = event.key;
        printf("SDL_KEYDOWN scancode=%d\n", e.keysym.scancode);
        switch (e.keysym.scancode) {
          case SDL_SCANCODE_GRAVE:
            frame_rate = 1;
            if (pause) nudge = true;
            else pause = true;
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
          case SDL_SCANCODE_TAB:
            draw_record = !draw_record;
            break;
          case SDL_SCANCODE_D:
            delete_agent = true;
            break;
          default:
            break;
        }
        break;
      }
    }
  }

  if (pause && ! nudge) return;
  nudge = false;

  for (int i = 0; i < FOOD_COUNT; i++) {
      if (foods[i].body->GetPosition().x < 0 || foods[i].body->GetPosition().x > WIDTH
      || foods[i].body->GetPosition().y < 0 || foods[i].body->GetPosition().y > HEIGHT) {
        init_food(&foods[i]);
      }
  }

  float32 timeStep = 1.0f / 60.0f;
  int32 velocityIterations = 6;
  int32 positionIterations = 2;
  world.Step(timeStep, velocityIterations, positionIterations);

  // food index maps every agent to its nearest food
  int food_index[NUM_AGENTS];
  for (int i = 0; i < NUM_AGENTS; i++) {
    float nearest_dist_sq;
    for (int j = 0; j < FOOD_COUNT; j++) {
      float dx = foods[j].body->GetPosition().x - agents[i].body->GetPosition().x;
      float dy = foods[j].body->GetPosition().y - agents[i].body->GetPosition().y;
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
    float dx = foods[food_index[i]].body->GetPosition().x - agents[i].body->GetPosition().x;
    float dy = foods[food_index[i]].body->GetPosition().y - agents[i].body->GetPosition().y;
    agent_inputs[i].nearest_food_relative_direction = angle_diff(atan2(dy, dx), agents[i].body->GetAngle());
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

    fog_flags[i] = false;

    //
    //
    // train via mouse
    int mouse_x, mouse_y;
    SDL_PumpEvents();
    if (SDL_GetMouseState(&mouse_x, &mouse_y) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      float dx_to_mouse = (float)mouse_x - agents[i].body->GetPosition().x;
      float dy_to_mouse = ((float)HEIGHT - (float)mouse_y) - agents[i].body->GetPosition().y;
      float fog_rotation = atan2(dy_to_mouse, dx_to_mouse);
      float delta_radians = angle_diff(fog_rotation, agents[i].body->GetAngle());
      float distance = sqrtf(dx_to_mouse * dx_to_mouse + dy_to_mouse * dy_to_mouse);
      printf("delta_radians=%f\ndistance=%f\n", delta_radians, distance);
      float fog_rotational_behavior = max(min(delta_radians, +1.0f), -1.0f);
      float fog_linear_behavior = max(min(distance, +1.0f), -1.0f);
      printf("fog_rotational_behavior=%f\nfog_linear_behavior=%f\n", fog_rotational_behavior, fog_linear_behavior);
      float ann_output_train[ANN_NUM_OUTPUT] = {
        fog_rotational_behavior,
        fog_linear_behavior
      };
      fann_train(agents[i].ann, ann_input, ann_output_train);
      printf("ann_output_train[0]=%f\nann_output_train[1]=%f\n", ann_output_train[0], ann_output_train[1]);
      fog_flags[i] = true;
      fog_behaviors[i].rotational = fog_rotational_behavior;
      fog_behaviors[i].linear = fog_linear_behavior;
    }

    float *ann_output = fann_run(agents[i].ann, ann_input);
    printf("ann_output[0]=%f\nann_output[1]=%f\n", ann_output[0], ann_output[1]);

    agent_behaviors[i].rotational = ann_output[0];
    agent_behaviors[i].linear = ann_output[1];
    printf("b.rotational=%f\nb.linear=%f\n", agent_behaviors[i].rotational, agent_behaviors[i].linear);

    float new_angle = agents[i].body->GetAngle() + agent_behaviors[i].rotational;
        float new_x = agents[i].body->GetPosition().x + agent_behaviors[i].linear * cos(new_angle);
        float new_y = agents[i].body->GetPosition().y + agent_behaviors[i].linear * sin(new_angle);
    agents[i].body->SetTransform(b2Vec2(new_x, new_y), new_angle);

    // decay health as a function of time
    agents[i].health -= HEALTH_DECAY_CONSTANT;
    if (delete_agent && i == selected_index) {
        delete_agent = false;
        agents[i].health = -1.0f;
    }

    // eat near food
    for (int j = 0; j < FOOD_COUNT; j++) {
       float dx = foods[j].body->GetPosition().x - agents[i].body->GetPosition().x;
       float dy = foods[j].body->GetPosition().y - agents[i].body->GetPosition().y;
       float dist_sq = (dx*dx) + (dy*dy);
       if(dist_sq < EAT_DISTANCE*EAT_DISTANCE) {
         agents[i].health = min(MAX_HEALTH, agents[i].health + FOOD_VALUE);
         agents[i].score++;
         init_food(&foods[j]);
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

      if (fdis(gen) < 0.9f && agents[selected_index].score > 0) {
        agents[i].parent_index = selected_index;
        // agents[i].x = agents[selected_index].x;
        // agents[i].y = agents[selected_index].y;
        fann *source_ann = agents[selected_index].ann;
        fann *target_ann = agents[i].ann;
        int num_conn = fann_get_total_connections(source_ann);
        fann_connection source_connections[num_conn];
        fann_connection target_connections[num_conn];
        fann_get_connection_array(source_ann, source_connections);
        fann_get_connection_array(target_ann, target_connections);
        for (int j = 0; j < num_conn; j++) {
          if (fdis(gen) < 0.3f) {
            target_connections[j].weight = source_connections[j].weight + ((fdis(gen) * 0.60f) - 0.30f);
          } else {
            target_connections[j].weight = source_connections[j].weight;
          }
        }
        fann_set_weight_array(target_ann, target_connections, num_conn);
      }
    }
  }

  // update the record model
  if (frame % RECORD_SAMPLE_RATE == 0) {
    records_index++;
    records_index %= WIDTH;
    fann *ann = agents[selected_index].ann;
    int num_conn = fann_get_total_connections(ann);
    fann_connection connections[num_conn];
    fann_get_connection_array(ann, connections);
    for (int i = 0; i < num_conn; i++) {
      records[records_index].weights[i] = connections[i].weight;
    }
    for (int i = 0; i < NUM_AGENTS; i++) {
        records[records_index].scores[i] = agents[i].score;
    }
    records[records_index].mse = fann_get_MSE(agents[selected_index].ann);
    records[records_index].rotational = agent_behaviors[selected_index].rotational;
    records[records_index].linear = agent_behaviors[selected_index].linear;
    records[records_index].fog_on = fog_flags[selected_index];
    records[records_index].fog_rotational = fog_behaviors[selected_index].rotational;
    records[records_index].fog_linear = fog_behaviors[selected_index].linear;
  }

  bool skip_render = (frame % frame_rate != 0);
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
      float r, g, b;
      hsv_to_rgb(agents[i].hue, 1.0f, 1.0f, &r, &g, &b);
      eg_set_color(r, g, b, 1.0f);
      eg_draw_line(agents[i].body->GetPosition().x,
                   agents[i].body->GetPosition().y,
                   agents[i].body->GetPosition().x + (float)cos(agents[i].body->GetAngle()) * 10.0F,
                   agents[i].body->GetPosition().y + (float)sin(agents[i].body->GetAngle()) * 10.0F,
                   10);


      // indicate nearest food (by food index)
      eg_set_color(0.9f, 0.9f, 0.9f, 0.3f);
      eg_draw_line(agents[i].body->GetPosition().x,
                   agents[i].body->GetPosition().y,
                   foods[food_index[i]].body->GetPosition().x,
                   foods[food_index[i]].body->GetPosition().y,
                   2);

      // agent
      float buddy_size = BUDDY_SIZE * (agents[i].parent_index == -1 ? 1.0f : 0.6f);
      if (i == selected_index) {
        float highlight_size = buddy_size * 1.2f;
        eg_set_color(0.99f, 0.99f, 0.01f, 0.9f);
        eg_draw_square(agents[selected_index].body->GetPosition().x - 0.5f*highlight_size, agents[selected_index].body->GetPosition().y - 0.5f*highlight_size, highlight_size, highlight_size);
      }
      eg_set_color(0.9f, 0.9f, 0.9f, 1.0f);
      eg_draw_square(agents[i].body->GetPosition().x - 0.5f*buddy_size, agents[i].body->GetPosition().y - 0.5f*buddy_size, buddy_size, buddy_size);
      eg_set_color(0.0f, 0.0f, 0.0f, 1.0f);
      eg_draw_square(agents[i].body->GetPosition().x - 0.5f*buddy_size*0.5f, agents[i].body->GetPosition().y - 0.5f*buddy_size*0.5f, buddy_size*0.5f, buddy_size*0.5f);

      // health bar
      eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
      eg_draw_square(agents[i].body->GetPosition().x - 15.0f, agents[i].body->GetPosition().y + 12.0f, 30.0f, 5.0f);
      if (agents[i].health > MAX_HEALTH * 0.25f) {
        eg_set_color(0.5f, 0.9f, 0.5f, 0.8f);
      } else {
        eg_set_color(0.8f, 0.3f, 0.3f, 0.8f);
      }
      eg_draw_square(agents[i].body->GetPosition().x - 15.0f, agents[i].body->GetPosition().y + 12.0f, agents[i].health * 30.0f / MAX_HEALTH, 5.0f);

      // score bar
      eg_set_color(0.2f, 0.2f, 0.2f, 0.7f);
      eg_draw_square(agents[i].body->GetPosition().x - 15.0f, agents[i].body->GetPosition().y + 20.0f, 30.0f, 5.0f);
      eg_set_color(0.9f, 0.85f, 0.0f, 0.8f);
      eg_draw_square(agents[i].body->GetPosition().x - 15.0f, agents[i].body->GetPosition().y + 20.0f, 30.0f * ((float)agents[i].score / (float)agents[high_score_index].score), 5.0f);
    }

    // draw foods
    for(int i = 0; i < FOOD_COUNT; i++) {
      eg_set_color(0.0f, 0.8f, 0.0f, 1.0f);
      eg_translate(foods[i].body->GetPosition().x, foods[i].body->GetPosition().y);
      eg_rotate(foods[i].body->GetAngle());
      eg_draw_square(-0.5f * FOOD_SIZE, -0.5 * FOOD_SIZE, FOOD_SIZE, FOOD_SIZE);
      eg_reset_transform();
    }

    // high score
    eg_set_color(0.9f, 0.3f, 0.3f, 1.0f);
    eg_draw_square(agents[high_score_index].body->GetPosition().x - 0.5f*BUDDY_SIZE, agents[high_score_index].body->GetPosition().y - 0.5f*BUDDY_SIZE, BUDDY_SIZE, BUDDY_SIZE);

    // record model
    if (draw_record) {
      // ann weight graph
      eg_set_color(0, 0, 0, 0.7f);
      eg_draw_square(0, 0, WIDTH, 100);
      for (int rx = 0; rx < WIDTH; rx++) {
        Record record = records[(records_index + WIDTH - rx) % WIDTH];
        float y = 101.0f;
        for (int wi = 0; wi < ANN_NUM_CONNECTIONS; wi++) {
          float r, g, b;
          float hue = (float)wi / (float)ANN_NUM_CONNECTIONS;
          while (hue > 1.0f) hue -= 1.0f;
          hsv_to_rgb(hue, 0.60f, 0.90f, &r, &g, &b);
          eg_set_color(r, g, b, 0.8f);
          float w = record.weights[wi];
          if (w < 0.0f) w *= -1;
          float h = w * 100.0f + 5.0f;
          eg_draw_line(rx, y, rx, y + h, 1.5f);
          y += h;
        }
        // total score graph
        for (int i = 0; i < NUM_AGENTS; ++i) {
          float r, g, b;
          hsv_to_rgb(agents[i].hue, 0.60f, 0.90f, &r, &g, &b);
          eg_set_color(r, g, b, 1.0f);
          eg_draw_square(rx, HEIGHT - record.scores[i], 1.0f, 1.0f);
        }
        // mse graph
        eg_set_color(1.0f, 0.0f, 0.0f, .9f);
        eg_draw_square(rx, record.mse * 100.0f, 1.0f, 1.0f);
        // rotational behavior
        eg_set_color(0.0f, 1.0f, 0.0f, .9f);
        eg_draw_square(rx, record.rotational * 50.0f + 50.0f, 1.0f, 1.0f);
        // linear behavior
        eg_set_color(0.0f, 0.0f, 1.0f, .9f);
        eg_draw_square(rx, record.linear * 50.0f + 50.0f, 1.0f, 1.0f);
        if (record.fog_on) {
          // fog rotational behavior
          eg_set_color(0.0f, 1.0f, 0.0f, .6f);
          eg_draw_square(rx, record.fog_rotational * 50.0f + 50.0f, 1.0f, 1.0f);
          // fog linear behavior
          eg_set_color(0.0f, 0.0f, 1.0f, .6f);
          eg_draw_square(rx, record.fog_linear * 50.0f + 50.0f, 1.0f, 1.0f);
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

  printf("%d\n", frame);

  eg_shutdown();

  return 0;
}
