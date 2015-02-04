#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <set>
#include "easygame.h"

using std::min;
using std::max;

const float DT = 1.0f/60.0f;
const float BUDDY_SIZE = 15.0f;
const float FOOD_SIZE = 8.0f;
const int FOOD_COUNT = 10;
const float FOOD_VALUE = 25.0f;
const float EAT_DISTANCE = 20.0f;
const float HEALTH_DECAY = 10.0f;
const float MAX_HEALTH = 100.0f;
const float PLAYER_MOVE_SPEED = 100.0f;

struct AgentInput {
  float nearest_food_dx;
  float nearest_food_dy;
};

struct AgentBehavior {
  float dx, dy;
};

struct Agent {
  float x, y;
  float health;
};

struct Food {
  float x, y;
};

Food make_food();

int main(int argc, char *argv[]) {
  eg_init(640, 480, "Buddies");

  Agent agents[2];
  Food foods[FOOD_COUNT];

  agents[0].x = 160.0f;
  agents[0].y = 240.0f;
  agents[0].health = MAX_HEALTH;

  agents[1].x = 480.0f;
  agents[1].y = 240.0f;
  agents[1].health = MAX_HEALTH;

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

    AgentInput agent_inputs[2];
    for(int i = 0; i < 2; i++) {
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

    AgentBehavior agent_behaviors[2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    if(eg_get_keystate(SDL_SCANCODE_LEFT))  agent_behaviors[0].dx -= PLAYER_MOVE_SPEED;
    if(eg_get_keystate(SDL_SCANCODE_RIGHT)) agent_behaviors[0].dx += PLAYER_MOVE_SPEED;
    if(eg_get_keystate(SDL_SCANCODE_DOWN))  agent_behaviors[0].dy -= PLAYER_MOVE_SPEED;
    if(eg_get_keystate(SDL_SCANCODE_UP))    agent_behaviors[0].dy += PLAYER_MOVE_SPEED;

    for(int i = 0; i < 2; i++) {
      agents[i].x += DT * agent_behaviors[i].dx;
      agents[i].y += DT * agent_behaviors[i].dy;
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
    }

    eg_clear_screen(0.7f, 0.8f, 0.75f, 0.0f);

    // draw agents
    float agent_colors[2][4] = {
      { 0.3f, 0.5f, 0.2f, 1.0f },
      { 0.3f, 0.2f, 0.5f, 1.0f }
    };

    for(int i = 0; i < 2; i++) {
      eg_set_color(agent_colors[i][0], agent_colors[i][1], agent_colors[i][2], agent_colors[i][3]);    
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
  Food f = { 640.0f * fdis(gen), 480.0f * fdis(gen) };
  return f;
}
