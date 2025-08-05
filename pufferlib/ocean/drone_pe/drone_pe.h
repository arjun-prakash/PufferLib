// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"
#include "dronelib.h"

#define TASK_PURSUIT_EVASION 0
#define TASK_N 1
#define DUMP_FRAMES 1
#define MAX_DUMP_FRAMES 300 
#define PI 3.14159265358979323846f

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
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int report_interval;

    int task;
    int num_agents;
    Drone* agents;

    Client *client;
} DronePE;

void init(DronePE *env) {
    env->agents = calloc(env->num_agents, sizeof(Drone));
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DronePE *env, int idx, bool oob) {
    Drone *agent = &env->agents[idx];
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

Drone* nearest_drone(DronePE* env, Drone *agent) {
    float min_dist = 999999.0f;
    Drone *nearest = NULL;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *other = &env->agents[i];
        if (other == agent) {
            continue;
        }
        float dx = agent->pos.x - other->pos.x;
        float dy = agent->pos.y - other->pos.y;
        float dz = agent->pos.z - other->pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }
    if (nearest == NULL) {
        int x = 0;

    }
    return nearest;
}

void compute_observations(DronePE *env) {
    int idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        Quat q_inv = quat_inverse(agent->quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, agent->vel);
        Vec3 drone_up_world = quat_rotate(agent->quat, (Vec3){0.0f, 0.0f, 1.0f});

        // TODO: Need abs observations now right?
        env->observations[idx++] = linear_vel_body.x / agent->max_vel;
        env->observations[idx++] = linear_vel_body.y / agent->max_vel;
        env->observations[idx++] = linear_vel_body.z / agent->max_vel;

        env->observations[idx++] = agent->omega.x / agent->max_omega;
        env->observations[idx++] = agent->omega.y / agent->max_omega;
        env->observations[idx++] = agent->omega.z / agent->max_omega;

        env->observations[idx++] = drone_up_world.x;
        env->observations[idx++] = drone_up_world.y;
        env->observations[idx++] = drone_up_world.z;

        env->observations[idx++] = agent->quat.w;
        env->observations[idx++] = agent->quat.x;
        env->observations[idx++] = agent->quat.y;
        env->observations[idx++] = agent->quat.z;

        env->observations[idx++] = agent->rpms[0] / agent->max_rpm;
        env->observations[idx++] = agent->rpms[1] / agent->max_rpm;
        env->observations[idx++] = agent->rpms[2] / agent->max_rpm;
        env->observations[idx++] = agent->rpms[3] / agent->max_rpm;

        env->observations[idx++] = agent->pos.x / GRID_X;
        env->observations[idx++] = agent->pos.y / GRID_Y;
        env->observations[idx++] = agent->pos.z / GRID_Z;

        env->observations[idx++] = agent->spawn_pos.x / GRID_X;
        env->observations[idx++] = agent->spawn_pos.y / GRID_Y;
        env->observations[idx++] = agent->spawn_pos.z / GRID_Z;

        float dx = agent->target_pos.x - agent->pos.x;
        float dy = agent->target_pos.y - agent->pos.y;
        float dz = agent->target_pos.z - agent->pos.z;
        env->observations[idx++] = clampf(dx, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dy, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dz, -1.0f, 1.0f);
        env->observations[idx++] = dx / GRID_X;
        env->observations[idx++] = dy / GRID_Y;
        env->observations[idx++] = dz / GRID_Z;

        env->observations[idx++] = agent->last_collision_reward;
        env->observations[idx++] = agent->last_target_reward;
        env->observations[idx++] = agent->last_abs_reward;

        // Multiagent obs
        Drone* nearest = nearest_drone(env, agent);
        if (env->num_agents > 1) {
            env->observations[idx++] = clampf(nearest->pos.x - agent->pos.x, -1.0f, 1.0f);
            env->observations[idx++] = clampf(nearest->pos.y - agent->pos.y, -1.0f, 1.0f);
            env->observations[idx++] = clampf(nearest->pos.z - agent->pos.z, -1.0f, 1.0f);
        } else {
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
        }
    }
}

void move_target(DronePE* env, Drone *agent) {
    agent->target_pos.x += agent->target_vel.x;
    agent->target_pos.y += agent->target_vel.y;
    agent->target_pos.z += agent->target_vel.z;
    if (agent->target_pos.x < -GRID_X || agent->target_pos.x > GRID_X) {
        agent->target_vel.x = -agent->target_vel.x;
    }
    if (agent->target_pos.y < -GRID_Y || agent->target_pos.y > GRID_Y) {
        agent->target_vel.y = -agent->target_vel.y;
    }
    if (agent->target_pos.z < -GRID_Z || agent->target_pos.z > GRID_Z) {
        agent->target_vel.z = -agent->target_vel.z;
    }
}

void set_target_pursuit_evasion(DronePE* env, int idx) {
    Drone* agent = &env->agents[idx];
    
    if (idx == 0) {
        // Evader: try to move away from closest pursuer
        float max_dist = 0.0f;
        Vec3 best_escape_pos = agent->pos;
        
        // Find the closest pursuer
        Drone* closest_pursuer = NULL;
        float min_pursuer_dist = FLT_MAX;
        for (int i = 1; i < env->num_agents; i++) {
            Drone* pursuer = &env->agents[i];
            float dx = agent->pos.x - pursuer->pos.x;
            float dy = agent->pos.y - pursuer->pos.y;
            float dz = agent->pos.z - pursuer->pos.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < min_pursuer_dist) {
                min_pursuer_dist = dist;
                closest_pursuer = pursuer;
            }
        }
        
        if (closest_pursuer != NULL) {
            // Set target in opposite direction from closest pursuer
            Vec3 escape_dir = {
                agent->pos.x - closest_pursuer->pos.x,
                agent->pos.y - closest_pursuer->pos.y,
                agent->pos.z - closest_pursuer->pos.z
            };
            float escape_dist = sqrtf(escape_dir.x*escape_dir.x + escape_dir.y*escape_dir.y + escape_dir.z*escape_dir.z);
            if (escape_dist > 0.1f) {
                escape_dir.x /= escape_dist;
                escape_dir.y /= escape_dist;
                escape_dir.z /= escape_dist;
                
                // Set target 5 units away in escape direction
                agent->target_pos = (Vec3){
                    agent->pos.x + escape_dir.x * 5.0f,
                    agent->pos.y + escape_dir.y * 5.0f,
                    agent->pos.z + escape_dir.z * 5.0f
                };
                
                // Clamp target to boundaries
                agent->target_pos.x = clampf(agent->target_pos.x, -GRID_X + 1.0f, GRID_X - 1.0f);
                agent->target_pos.y = clampf(agent->target_pos.y, -GRID_Y + 1.0f, GRID_Y - 1.0f);
                agent->target_pos.z = clampf(agent->target_pos.z, -GRID_Z + 1.0f, GRID_Z - 1.0f);
            } else {
                agent->target_pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), rndf(-MARGIN_Z, MARGIN_Z)};
            }
        } else {
            agent->target_pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), rndf(-MARGIN_Z, MARGIN_Z)};
        }
        agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
    } else {
        // Pursuer: target the evader's position
        Drone* evader = &env->agents[0];
        agent->target_pos = evader->pos;
        agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
    }
}

void set_target(DronePE* env, int idx) {
    set_target_pursuit_evasion(env, idx);
}

float compute_reward(DronePE* env, Drone *agent, bool collision) {
    float dist_reward = 0.0f;
    
    // Find agent index
    int agent_idx = -1;
    for (int i = 0; i < env->num_agents; i++) {
        if (&env->agents[i] == agent) {
            agent_idx = i;
            break;
        }
    }
    
    if (agent_idx == 0) {
        // Evader: reward for maximizing distance to closest pursuer
        float min_pursuer_dist = FLT_MAX;
        for (int i = 1; i < env->num_agents; i++) {
            Drone* pursuer = &env->agents[i];
            float dx = agent->pos.x - pursuer->pos.x;
            float dy = agent->pos.y - pursuer->pos.y;
            float dz = agent->pos.z - pursuer->pos.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < min_pursuer_dist) {
                min_pursuer_dist = dist;
            }
        }
        // Normalize distance reward for evader (higher distance = higher reward)
        dist_reward = clampf(min_pursuer_dist / 15.0f, 0.0f, 1.0f);
    } else {
        // Pursuer: reward for minimizing distance to evader
        Drone* evader = &env->agents[0];
        float dx = agent->pos.x - evader->pos.x;
        float dy = agent->pos.y - evader->pos.y;
        float dz = agent->pos.z - evader->pos.z;
        float dist_to_evader = sqrtf(dx*dx + dy*dy + dz*dz);
        // Normalize distance reward for pursuer (lower distance = higher reward)
        dist_reward = 1.0f - clampf(dist_to_evader / 15.0f, 0.0f, 1.0f);
    }

    // Collision avoidance: only penalize same-type agents getting too close
    float density_reward = 0.0f;
    if (collision && env->num_agents > 1) {
        if (agent_idx == 0) {
            // Evader: avoid other evaders (if multiple exist)
            for (int i = 1; i < env->num_agents; i++) {
                if (env->agents[i].pos.x == env->agents[0].pos.x) continue; // Skip if same as evader
                float dx = agent->pos.x - env->agents[i].pos.x;
                float dy = agent->pos.y - env->agents[i].pos.y;
                float dz = agent->pos.z - env->agents[i].pos.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < 2.0f && i != 0) {
                    // Only penalize evader-evader collisions (if there were multiple evaders)
                    // For now, this won't trigger since we only have one evader
                }
            }
        } else {
            // Pursuer: avoid other pursuers
            for (int i = 1; i < env->num_agents; i++) {
                if (i == agent_idx) continue; // Skip self
                Drone* other_pursuer = &env->agents[i];
                float dx = agent->pos.x - other_pursuer->pos.x;
                float dy = agent->pos.y - other_pursuer->pos.y;
                float dz = agent->pos.z - other_pursuer->pos.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < 1.5f) {
                    density_reward = -0.3f; // Reduced penalty for pursuer-pursuer collisions
                    agent->collisions += 1.0f;
                    break; // Only count one collision per step
                }
            }
        }
    }

    float abs_reward = dist_reward + density_reward;

    // Prevent negative dist and density from making a positive reward
    if (dist_reward < 0.0f && density_reward < 0.0f) {
        abs_reward *= -1.0f;
    }

    float delta_reward = abs_reward - agent->last_abs_reward;

    agent->last_collision_reward = density_reward;
    agent->last_target_reward = dist_reward;
    agent->last_abs_reward = abs_reward;

    agent->episode_length++;
    agent->score += abs_reward;

    return delta_reward;
}

void reset_agent(DronePE* env, Drone *agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    
    // Intelligent spawning with minimum distance requirements
    Vec3 spawn_pos;
    bool valid_position = false;
    int max_attempts = 50;
    int attempts = 0;
    
    while (!valid_position && attempts < max_attempts) {
        spawn_pos = (Vec3){rndf(-8, 8), rndf(-8, 8), rndf(-8, 8)};
        valid_position = true;
        
        // Check distance to other agents
        for (int i = 0; i < idx; i++) {
            Drone* other = &env->agents[i];
            float dx = spawn_pos.x - other->pos.x;
            float dy = spawn_pos.y - other->pos.y;
            float dz = spawn_pos.z - other->pos.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            
            // Minimum distance requirements
            float min_dist = 3.0f; // Default minimum distance
            if (idx == 0 || i == 0) {
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
        if (idx == 0) {
            // Evader spawns in center area
            spawn_pos = (Vec3){rndf(-2, 2), rndf(-2, 2), rndf(-2, 2)};
        } else {
            // Pursuers spawn around the edges
            float angle = (float)(idx - 1) / (float)(env->num_agents - 1) * 2.0f * PI;
            float radius = 8.0f;
            spawn_pos = (Vec3){
                radius * cosf(angle) + rndf(-1, 1),
                radius * sinf(angle) + rndf(-1, 1),
                rndf(-8, 8)
            };
        }
    }
    
    agent->pos = spawn_pos;
    agent->spawn_pos = agent->pos;
    agent->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->omega = (Vec3){0.0f, 0.0f, 0.0f};
    agent->quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};

    float size = rndf(0.1f, 0.4);
    init_drone(agent, size, 0.1f);
    compute_reward(env, agent, false); // Start without collision checking
}

void c_reset(DronePE *env) {
    env->tick = 0;
    env->task = TASK_PURSUIT_EVASION;

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env, i);
    }

    compute_observations(env);
}

void c_step(DronePE *env) {
    env->tick = (env->tick + 1) % HORIZON;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;

        float* atn = &env->actions[4*i];
        move_drone(agent, atn);

        // check out of bounds
        bool out_of_bounds = agent->pos.x < -GRID_X || agent->pos.x > GRID_X ||
                             agent->pos.y < -GRID_Y || agent->pos.y > GRID_Y ||
                             agent->pos.z < -GRID_Z || agent->pos.z > GRID_Z;

        move_target(env, agent);

        // Compute pursuit-evasion reward with appropriate collision detection
        // Only check collisions for pursuers (to avoid each other) 
        bool check_collisions = (i > 0); // Only pursuers need collision avoidance
        float reward = compute_reward(env, agent, check_collisions);

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

void c_close(DronePE *env) {
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

Client *make_client(DronePE *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib DronePE");

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
    client->camera_elevation = PI / 10.0f;
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

void c_render(DronePE *env) {
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
        Drone *agent = &env->agents[i];
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

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f,
        GRID_Y * 2.0f, GRID_Z * 2.0f, WHITE);

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        // draws drone body with pursuit-evasion colors
        Color body_color = PUFF_CYAN; // Default color
        if (i == 0) {
            body_color = GREEN;  // Evader is green
        } else {
            body_color = RED;    // Pursuers are red
        }
        DrawSphere((Vector3){agent->pos.x, agent->pos.y, agent->pos.z}, 0.3f, body_color);

        // draws rotors according to thrust
        float T[4];
        for (int j = 0; j < 4; j++) {
            float rpm = (env->actions[4*i + j] + 1.0f) * 0.5f * agent->max_rpm;
            T[j] = agent->k_thrust * rpm * rpm;
        }

        const float rotor_radius = 0.15f;
        const float visual_arm_len = agent->arm_len * 4.0f;

        Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                      {-visual_arm_len, 0.0f, 0.0f},
                                      {0.0f, +visual_arm_len, 0.0f},
                                      {0.0f, -visual_arm_len, 0.0f}};

        Color base_colors[4] = {body_color, body_color, body_color, body_color};

        for (int j = 0; j < 4; j++) {
            Vec3 world_off = quat_rotate(agent->quat, rotor_offsets_body[j]);

            Vector3 rotor_pos = {agent->pos.x + world_off.x, agent->pos.y + world_off.y,
                                 agent->pos.z + world_off.z};

            float rpm = (env->actions[4*i + j] + 1.0f) * 0.5f * agent->max_rpm;
            float intensity = 0.75f + 0.25f * (rpm / agent->max_rpm);

            Color rotor_color = (Color){(unsigned char)(base_colors[j].r * intensity),
                                        (unsigned char)(base_colors[j].g * intensity),
                                        (unsigned char)(base_colors[j].b * intensity), 255};

            DrawSphere(rotor_pos, rotor_radius, rotor_color);

            DrawCylinderEx((Vector3){agent->pos.x, agent->pos.y, agent->pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                           BLACK);
        }

        // draws line with direction and magnitude of velocity / 10
        if (norm3(agent->vel) > 0.1f) {
            DrawLine3D((Vector3){agent->pos.x, agent->pos.y, agent->pos.z},
                       (Vector3){agent->pos.x + agent->vel.x * 0.1f, agent->pos.y + agent->vel.y * 0.1f,
                                 agent->pos.z + agent->vel.z * 0.1f},
                       MAGENTA);
        }

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
            if (i == 0) {
                base_trail_color = (Color){0, 255, 0, 255}; // Green for evader
            } else {
                base_trail_color = (Color){255, 0, 0, 255}; // Red for pursuers
            }
            Color trail_color = ColorAlpha(base_trail_color, alpha);
            
            DrawLine3D((Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
                       (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
                       trail_color);
        }
    }

    if (IsKeyDown(KEY_TAB)) {
        for (int i = 0; i < env->num_agents; i++) {
            Drone *agent = &env->agents[i];
            Vec3 target_pos = agent->target_pos;
            DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.45f, (Color){0, 255, 255, 100});
        }
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);
    DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, 50, 16, PUFF_WHITE);
    
    // Show pursuit-evasion status
    if (env->num_agents > 1) {
        Drone* evader = &env->agents[0];
        float min_distance = FLT_MAX;
        for (int i = 1; i < env->num_agents; i++) {
            Drone* pursuer = &env->agents[i];
            float dx = evader->pos.x - pursuer->pos.x;
            float dy = evader->pos.y - pursuer->pos.y;  
            float dz = evader->pos.z - pursuer->pos.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
        DrawText(TextFormat("Closest Pursuer Distance: %.2f", min_distance), 10, 70, 16, PUFF_WHITE);
        DrawText(TextFormat("Agents: 1 Evader (Green), %d Pursuers (Red)", env->num_agents - 1), 10, 90, 16, PUFF_WHITE);
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
