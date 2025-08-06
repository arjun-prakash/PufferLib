// Originally adapted from drone implementation by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50
#define HORIZON 1024

// Physical constants for Dubins cars
#define BASE_SPEED_EVADER 1.0f        // Evader speed (slower)
#define BASE_SPEED_PURSUER 1.3f       // Pursuer speed (faster)
#define TURN_ANGLE (30.0f * PI / 180.0f)  // 30 degrees in radians
#define PI 3.14159265358979323846f

// Simulation properties
#define GRID_X 30.0f
#define GRID_Y 30.0f
#define MARGIN_X (GRID_X - 1)
#define MARGIN_Y (GRID_Y - 1)
#define DT 0.05f
#define DT_RNG 0.0f

// Corner to corner distance
#define MAX_DIST sqrtf((2*GRID_X)*(2*GRID_X) + (2*GRID_Y)*(2*GRID_Y))

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float collision_rate;
    float oob;
    float score;
    float perf;
    float n;
};

typedef struct {
    float x, y;
} Vec2;

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec2 add2(Vec2 a, Vec2 b) { 
    return (Vec2){a.x + b.x, a.y + b.y}; 
}

static inline Vec2 sub2(Vec2 a, Vec2 b) { 
    return (Vec2){a.x - b.x, a.y - b.y}; 
}

static inline Vec2 scalmul2(Vec2 a, float b) { 
    return (Vec2){a.x * b, a.y * b}; 
}

static inline float dot2(Vec2 a, Vec2 b) { 
    return a.x * b.x + a.y * b.y; 
}

static inline float norm2(Vec2 a) { 
    return sqrtf(dot2(a, a)); 
}

static inline void clamp2(Vec2 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
}

// Normalize angle to [-PI, PI]
static inline float normalize_angle(float angle) {
    while (angle > PI) angle -= 2.0f * PI;
    while (angle < -PI) angle += 2.0f * PI;
    return angle;
}

typedef struct {
    Vec2 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec2 spawn_pos;
    Vec2 pos;           // position (x, y)
    Vec2 prev_pos;      // previous position for collision detection
    float heading;      // heading angle in radians
    float speed;        // forward speed
    
    Vec2 target_pos;
    Vec2 target_vel;
   
    float last_abs_reward;
    float last_target_reward;
    float last_collision_reward;
    float episode_return;
    float collisions;
    int episode_length;
    float score;

    // Car properties
    bool is_evader;     // true if evader, false if pursuer
    float max_speed;    // maximum speed
} DubinsCar;

void init_dubins_car(DubinsCar* car, bool is_evader) {
    car->is_evader = is_evader;
    
    if (is_evader) {
        car->max_speed = BASE_SPEED_EVADER;
        car->speed = BASE_SPEED_EVADER;
    } else {
        car->max_speed = BASE_SPEED_PURSUER;
        car->speed = BASE_SPEED_PURSUER;
    }
    
    car->heading = 0.0f;
}

void move_dubins_car(DubinsCar* car, int action) {
    // Actions: 0 = turn left, 1 = straight, 2 = turn right
    // Clamp action to valid range
    if (action < 0) action = 0;
    if (action > 2) action = 2;
    
    car->prev_pos = car->pos;
    
    // Update heading based on action
    switch (action) {
        case 0: // Turn left
            car->heading += TURN_ANGLE;
            break;
        case 1: // Go straight
            // No heading change
            break;
        case 2: // Turn right
            car->heading -= TURN_ANGLE;
            break;
    }
    
    // Normalize heading
    car->heading = normalize_angle(car->heading);
    
    // Update position based on heading and speed
    float dt = DT * rndf(1.0f - DT_RNG, 1.0f + DT_RNG);
    car->pos.x += car->speed * cosf(car->heading) * dt;
    car->pos.y += car->speed * sinf(car->heading) * dt;
}
