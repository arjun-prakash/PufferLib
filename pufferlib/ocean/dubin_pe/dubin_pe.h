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
#include "dubinlib.h"

#define TASK_PURSUIT_EVASION 0
#define TASK_N 1
#define DUMP_FRAMES 1
#define MAX_DUMP_FRAMES 300

char* TASK_NAMES[TASK_N] = {
    "Pursuit-Evasion"
};

typedef struct Client Client;
struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;

    // Trailing path buffer (for rendering only)
    Trail* trails;
};

typedef struct {
    float *observations;
    int *actions;        // Changed to int for discrete actions
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int report_interval;

    int task;
    int num_agents;
    int num_pursuers;
    DubinsCar* agents;

    // Configuration parameters
    float evader_speed;
    float pursuer_speed;
    float turning_angle_deg;

    Client *client;
} DubinPE;

void init(DubinPE *env) {
    env->agents = calloc(env->num_agents, sizeof(DubinsCar));
    env->log = (Log){0};
    env->tick = 0;
    
    // Set default values if not already set
    if (env->num_pursuers == 0) env->num_pursuers = env->num_agents - 1; // Default: all except first are pursuers
    if (env->evader_speed == 0.0f) env->evader_speed = DEFAULT_SPEED_EVADER;
    if (env->pursuer_speed == 0.0f) env->pursuer_speed = DEFAULT_SPEED_PURSUER;
    if (env->turning_angle_deg == 0.0f) env->turning_angle_deg = DEFAULT_TURN_ANGLE * 180.0f / PI;
}

void add_log(DubinPE *env, int idx, bool oob) {
    DubinsCar *agent = &env->agents[idx];
    env->log.score += agent->score;
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collision_rate += agent->collisions / (float)agent->episode_length;
    env->log.perf += agent->score / (float)agent->episode_length;
    if (oob) {
        env->log.oob += 1.0f;
    }
    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
}

DubinsCar* nearest_car(DubinPE* env, DubinsCar *agent) {
    float min_dist = 999999.0f;
    DubinsCar *nearest = NULL;
    for (int i = 0; i < env->num_agents; i++) {
        DubinsCar *other = &env->agents[i];
        if (other == agent) {
            continue;
        }
        float dx = agent->pos.x - other->pos.x;
        float dy = agent->pos.y - other->pos.y;
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }
    return nearest;
}

void compute_observations(DubinPE *env) {
    int idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        DubinsCar *agent = &env->agents[i];

        // Position normalized
        env->observations[idx++] = agent->pos.x / GRID_X;
        env->observations[idx++] = agent->pos.y / GRID_Y;

        // Heading normalized
        env->observations[idx++] = agent->heading / PI;

        // Speed normalized
        env->observations[idx++] = agent->speed / agent->max_speed;

        // Spawn position normalized
        env->observations[idx++] = agent->spawn_pos.x / GRID_X;
        env->observations[idx++] = agent->spawn_pos.y / GRID_Y;

        // Target position relative
        float dx = agent->target_pos.x - agent->pos.x;
        float dy = agent->target_pos.y - agent->pos.y;
        env->observations[idx++] = clampf(dx, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dy, -1.0f, 1.0f);
        env->observations[idx++] = dx / GRID_X;
        env->observations[idx++] = dy / GRID_Y;

        // Reward history
        env->observations[idx++] = agent->last_collision_reward;
        env->observations[idx++] = agent->last_target_reward;
        env->observations[idx++] = agent->last_abs_reward;

        // Agent type
        env->observations[idx++] = agent->is_evader ? 1.0f : -1.0f;

        // Nearest agent info
        DubinsCar* nearest = nearest_car(env, agent);
        if (env->num_agents > 1 && nearest != NULL) {
            env->observations[idx++] = clampf(nearest->pos.x - agent->pos.x, -1.0f, 1.0f);
            env->observations[idx++] = clampf(nearest->pos.y - agent->pos.y, -1.0f, 1.0f);
            env->observations[idx++] = nearest->heading / PI;
        } else {
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
        }
    }
}

void move_target(DubinPE* env, DubinsCar *agent) {
    agent->target_pos.x += agent->target_vel.x;
    agent->target_pos.y += agent->target_vel.y;
    if (agent->target_pos.x < -GRID_X || agent->target_pos.x > GRID_X) {
        agent->target_vel.x = -agent->target_vel.x;
    }
    if (agent->target_pos.y < -GRID_Y || agent->target_pos.y > GRID_Y) {
        agent->target_vel.y = -agent->target_vel.y;
    }
}

void set_target_pursuit_evasion(DubinPE* env, int idx) {
    DubinsCar* agent = &env->agents[idx];
    
    if (agent->is_evader) {
        // Evader: try to move away from closest pursuer
        DubinsCar* closest_pursuer = NULL;
        float min_pursuer_dist = FLT_MAX;
        for (int i = 0; i < env->num_agents; i++) {
            DubinsCar* pursuer = &env->agents[i];
            if (pursuer->is_evader) continue; // Skip other evaders
            
            float dx = agent->pos.x - pursuer->pos.x;
            float dy = agent->pos.y - pursuer->pos.y;
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < min_pursuer_dist) {
                min_pursuer_dist = dist;
                closest_pursuer = pursuer;
            }
        }
        
        if (closest_pursuer != NULL) {
            // Set target in opposite direction from closest pursuer
            Vec2 escape_dir = {
                agent->pos.x - closest_pursuer->pos.x,
                agent->pos.y - closest_pursuer->pos.y
            };
            float escape_dist = sqrtf(escape_dir.x*escape_dir.x + escape_dir.y*escape_dir.y);
            if (escape_dist > 0.1f) {
                escape_dir.x /= escape_dist;
                escape_dir.y /= escape_dist;
                
                // Set target 5 units away in escape direction
                agent->target_pos = (Vec2){
                    agent->pos.x + escape_dir.x * 5.0f,
                    agent->pos.y + escape_dir.y * 5.0f
                };
                
                // Clamp target to boundaries
                agent->target_pos.x = clampf(agent->target_pos.x, -GRID_X + 1.0f, GRID_X - 1.0f);
                agent->target_pos.y = clampf(agent->target_pos.y, -GRID_Y + 1.0f, GRID_Y - 1.0f);
            } else {
                agent->target_pos = (Vec2){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y)};
            }
        } else {
            agent->target_pos = (Vec2){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y)};
        }
        agent->target_vel = (Vec2){0.0f, 0.0f};
    } else {
        // Pursuer: target the nearest evader's position
        DubinsCar* nearest_evader = NULL;
        float min_evader_dist = FLT_MAX;
        for (int i = 0; i < env->num_agents; i++) {
            DubinsCar* evader = &env->agents[i];
            if (!evader->is_evader) continue; // Skip pursuers
            
            float dx = agent->pos.x - evader->pos.x;
            float dy = agent->pos.y - evader->pos.y;
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < min_evader_dist) {
                min_evader_dist = dist;
                nearest_evader = evader;
            }
        }
        
        if (nearest_evader != NULL) {
            agent->target_pos = nearest_evader->pos;
        } else {
            agent->target_pos = (Vec2){0.0f, 0.0f}; // Center if no evader found
        }
        agent->target_vel = (Vec2){0.0f, 0.0f};
    }
}

void set_target(DubinPE* env, int idx) {
    set_target_pursuit_evasion(env, idx);
}

float compute_reward(DubinPE* env, DubinsCar *agent, bool collision) {
    // Compute the average distance between evaders and pursuers for zero-sum reward
    float total_distance = 0.0f;
    int distance_count = 0;
    
    for (int i = 0; i < env->num_agents; i++) {
        for (int j = i + 1; j < env->num_agents; j++) {
            DubinsCar* agent1 = &env->agents[i];
            DubinsCar* agent2 = &env->agents[j];
            
            // Only count distances between evader and pursuer pairs
            if (agent1->is_evader != agent2->is_evader) {
                float dx = agent1->pos.x - agent2->pos.x;
                float dy = agent1->pos.y - agent2->pos.y;
                float dist = sqrtf(dx*dx + dy*dy);
                total_distance += dist;
                distance_count++;
            }
        }
    }
    
    float avg_distance = (distance_count > 0) ? total_distance / distance_count : 0.0f;
    float normalized_distance = clampf(avg_distance / 15.0f, 0.0f, 1.0f);
    
    // Zero-sum: evaders get positive reward for larger distances, pursuers get negative
    float dist_reward = agent->is_evader ? normalized_distance : -normalized_distance;

    // Collision avoidance: penalize same-type agents getting too close
    float density_reward = 0.0f;
    if (collision && env->num_agents > 1) {
        for (int i = 0; i < env->num_agents; i++) {
            if (i == (agent - env->agents)) continue; // Skip self
            DubinsCar* other = &env->agents[i];
            
            // Only penalize collisions between same types
            if (agent->is_evader == other->is_evader) {
                float dx = agent->pos.x - other->pos.x;
                float dy = agent->pos.y - other->pos.y;
                float dist = sqrtf(dx*dx + dy*dy);
                if (dist < 1.5f) {
                    density_reward = -0.3f;
                    agent->collisions += 1.0f;
                    break; // Only count one collision per step
                }
            }
        }
    }

    float abs_reward = dist_reward + density_reward;

    float delta_reward = abs_reward - agent->last_abs_reward;

    agent->last_collision_reward = density_reward;
    agent->last_target_reward = dist_reward;
    agent->last_abs_reward = abs_reward;

    agent->episode_length++;
    agent->score += abs_reward;

    return delta_reward;
}

void reset_agent(DubinPE* env, DubinsCar *agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    
    // Determine if this is an evader (first agent) or pursuer
    // First agent is always evader, rest are pursuers
    bool is_evader = (idx == 0);
    
    // Intelligent spawning with minimum distance requirements
    Vec2 spawn_pos;
    bool valid_position = false;
    int max_attempts = 50;
    int attempts = 0;
    
    while (!valid_position && attempts < max_attempts) {
        spawn_pos = (Vec2){rndf(-8, 8), rndf(-8, 8)};
        valid_position = true;
        
        // Check distance to other agents
        for (int i = 0; i < idx; i++) {
            DubinsCar* other = &env->agents[i];
            float dx = spawn_pos.x - other->pos.x;
            float dy = spawn_pos.y - other->pos.y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            // Minimum distance requirements
            float min_dist = 3.0f; // Default minimum distance
            if (is_evader || env->agents[i].is_evader) {
                // Evader-pursuer spacing should be larger
                min_dist = 5.0f;
            } else {
                // Pursuer-pursuer spacing can be smaller
                min_dist = 3.0f;
            }
            
            if (dist < min_dist) {
                valid_position = false;
                break;
            }
        }
        attempts++;
    }
    
    // If we couldn't find a valid position after max attempts, use a fallback
    if (!valid_position) {
        if (is_evader) {
            // Evader spawns in center area
            spawn_pos = (Vec2){rndf(-2, 2), rndf(-2, 2)};
        } else {
            // Pursuers spawn around the edges
            float angle = (float)(idx - 1) / (float)(env->num_agents - 1) * 2.0f * PI;
            float radius = 8.0f;
            spawn_pos = (Vec2){
                radius * cosf(angle) + rndf(-1, 1),
                radius * sinf(angle) + rndf(-1, 1)
            };
        }
    }
    
    agent->pos = spawn_pos;
    agent->spawn_pos = agent->pos;
    agent->heading = rndf(0, 2.0f * PI);

    init_dubins_car(agent, is_evader, env->evader_speed, env->pursuer_speed);
    compute_reward(env, agent, false); // Start without collision checking
}

void c_reset(DubinPE *env) {
    env->tick = 0;
    env->task = TASK_PURSUIT_EVASION;

    for (int i = 0; i < env->num_agents; i++) {
        DubinsCar *agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env, i);
    }

    compute_observations(env);
}

void c_step(DubinPE *env) {
    env->tick = (env->tick + 1) % HORIZON;
    for (int i = 0; i < env->num_agents; i++) {
        DubinsCar *agent = &env->agents[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;

        int action = env->actions[i];
        float turn_angle_rad = env->turning_angle_deg * PI / 180.0f;
        move_dubins_car(agent, action, turn_angle_rad);

        // check out of bounds
        bool out_of_bounds = agent->pos.x < -GRID_X || agent->pos.x > GRID_X ||
                             agent->pos.y < -GRID_Y || agent->pos.y > GRID_Y;

        move_target(env, agent);

        // Compute pursuit-evasion reward with collision detection
        float reward = compute_reward(env, agent, true);

        env->rewards[i] += reward;
        agent->episode_return += reward;

        if (out_of_bounds) {
            env->rewards[i] -= 1;
            env->terminals[i] = 1;
            add_log(env, i, true);
            reset_agent(env, agent, i);
        } else if (env->tick >= HORIZON - 1) {
            env->terminals[i] = 1;
            add_log(env, i, false);
        }
    }
    if (env->tick >= HORIZON - 1) {
        c_reset(env);
    }

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(DubinPE *env) {
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

Client *make_client(DubinPE *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib DubinPE");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 40.0f;
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 3.0f; // Higher elevation for top-down view
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    // Initialize trail buffer
    client->trails = (Trail*)calloc(env->num_agents, sizeof(Trail));
    for (int i = 0; i < env->num_agents; i++) {
        Trail* trail = &client->trails[i];
        trail->index = 0;
        trail->count = 0;
        for (int j = 0; j < TRAIL_LENGTH; j++) {
            trail->pos[j] = env->agents[i].pos;
        }
    }

    return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(DubinPE *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    handle_camera_controls(env->client);

    Client *client = env->client;

    for (int i = 0; i < env->num_agents; i++) {
        DubinsCar *agent = &env->agents[i];
        Trail *trail = &client->trails[i];
        trail->pos[trail->index] = agent->pos;
        trail->index = (trail->index + 1) % TRAIL_LENGTH;
        if (trail->count < TRAIL_LENGTH) {
            trail->count++;
        }
        if (env->terminals[i]) {
            trail->index = 0;
            trail->count = 0;
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    BeginMode3D(client->camera);

    // Draw ground plane
    DrawPlane((Vector3){0.0f, 0.0f, 0.0f}, (Vector2){GRID_X * 2.0f, GRID_Y * 2.0f}, DARKGRAY);
    
    // Draw bounding rectangle
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f,
        GRID_Y * 2.0f, 0.1f, WHITE);

    for (int i = 0; i < env->num_agents; i++) {
        DubinsCar *agent = &env->agents[i];

        // Draw car body with pursuit-evasion colors
        Color body_color;
        if (agent->is_evader) {
            body_color = GREEN;  // Evader is green
        } else {
            body_color = RED;    // Pursuers are red
        }
        
        // Draw car as a cylinder (represents the car body)
        DrawCylinder((Vector3){agent->pos.x, agent->pos.y, 0.1f}, 0.1f, 0.1f, 0.3f, 8, body_color);
        
        // Draw heading indicator (line showing direction)
        float line_length = 0.7f;
        Vector3 start = {agent->pos.x, agent->pos.y, 0.2f};
        Vector3 end = {
            agent->pos.x + line_length * cosf(agent->heading),
            agent->pos.y + line_length * sinf(agent->heading),
            0.2f
        };
        DrawLine3D(start, end, body_color);
        
        // Draw arrow head
        float arrow_size = 0.2f;
        Vector3 arrow_left = {
            end.x - arrow_size * cosf(agent->heading + PI * 0.8f),
            end.y - arrow_size * sinf(agent->heading + PI * 0.8f),
            0.2f
        };
        Vector3 arrow_right = {
            end.x - arrow_size * cosf(agent->heading - PI * 0.8f),
            end.y - arrow_size * sinf(agent->heading - PI * 0.8f),
            0.2f
        };
        DrawLine3D(end, arrow_left, body_color);
        DrawLine3D(end, arrow_right, body_color);

        // Draw trailing path
        Trail *trail = &client->trails[i];
        if (trail->count <= 2) {
            continue;
        }
        for (int j = 0; j < trail->count - 1; j++) {
            int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
            float alpha = (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
            
            // Different trail colors for evaders vs pursuers
            Color base_trail_color;
            if (agent->is_evader) {
                base_trail_color = (Color){0, 255, 0, 255}; // Green for evader
            } else {
                base_trail_color = (Color){255, 0, 0, 255}; // Red for pursuers
            }
            Color trail_color = ColorAlpha(base_trail_color, alpha);
            
            DrawLine3D((Vector3){trail->pos[idx0].x, trail->pos[idx0].y, 0.05f},
                       (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, 0.05f},
                       trail_color);
        }
    }

    if (IsKeyDown(KEY_TAB)) {
        for (int i = 0; i < env->num_agents; i++) {
            DubinsCar *agent = &env->agents[i];
            Vec2 target_pos = agent->target_pos;
            DrawSphere((Vector3){target_pos.x, target_pos.y, 0.1f}, 0.45f, (Color){0, 255, 255, 100});
        }
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);
    DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, 50, 16, PUFF_WHITE);
    
    // Show pursuit-evasion status
    if (env->num_agents > 1) {
        DubinsCar* evader = NULL;
        // Find the evader
        for (int i = 0; i < env->num_agents; i++) {
            if (env->agents[i].is_evader) {
                evader = &env->agents[i];
                break;
            }
        }
        
        if (evader != NULL) {
            float min_distance = FLT_MAX;
            int pursuer_count = 0;
            for (int i = 0; i < env->num_agents; i++) {
                DubinsCar* pursuer = &env->agents[i];
                if (!pursuer->is_evader) {
                    pursuer_count++;
                    float dx = evader->pos.x - pursuer->pos.x;
                    float dy = evader->pos.y - pursuer->pos.y;  
                    float dist = sqrtf(dx*dx + dy*dy);
                    if (dist < min_distance) {
                        min_distance = dist;
                    }
                }
            }
            DrawText(TextFormat("Closest Pursuer Distance: %.2f", min_distance), 10, 70, 16, PUFF_WHITE);
            DrawText(TextFormat("Agents: 1 Evader (Green), %d Pursuers (Red)", pursuer_count), 10, 90, 16, PUFF_WHITE);
        }
    }

    EndDrawing();

    #if DUMP_FRAMES
    static int frame_counter = 0;
    if (frame_counter < MAX_DUMP_FRAMES) {
        char fname[256];
        sprintf(fname, "frame_%06d.png", frame_counter);
        TakeScreenshot(fname);  // directly writes the PNG of the current framebuffer
        frame_counter++;
    }
    #endif
}
