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

typedef struct Client Client;

#define DUMP_FRAMES 1
#define MAX_DUMP_FRAMES 300 

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
    Vec3 trail[TRAIL_LENGTH];
    int trail_index;
    int trail_count;
};

typedef struct DroneCrazyflie DroneCrazyflie;
struct DroneCrazyflie {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int report_interval;
    int score;
    float episodic_return;

    int max_moves;
    int moves_left;

    Drone drone;
    Client *client;
};

void init(DroneCrazyflie *env) {
    env->log = (Log){0};
    env->tick = 0;

}

void add_log(DroneCrazyflie *env) {
    env->log.score += env->score;
    env->log.episode_return += env->episodic_return;
    env->log.episode_length += env->tick;
    env->log.perf += 1.0f;
    env->log.n += 1.0f;
}

void compute_observations(DroneCrazyflie *env) {
    Drone *drone = &env->drone;

    // Target is the origin (0, 0, 0)
    Vec3 target_pos = {0.0f, 0.0f, 0.0f};
    
    Quat q_inv = quat_inverse(drone->quat);
    Vec3 to_target = quat_rotate(q_inv, sub3(target_pos, drone->pos));
    Vec3 linear_vel_body = quat_rotate(q_inv, drone->vel);
    Vec3 drone_up_world = quat_rotate(drone->quat, (Vec3){0.0f, 0.0f, 1.0f});

    // Position relative to origin (target)
    env->observations[0] = to_target.x / GRID_SIZE;
    env->observations[1] = to_target.y / GRID_SIZE;
    env->observations[2] = to_target.z / GRID_SIZE;

    // Linear velocity in body frame
    env->observations[3] = linear_vel_body.x / drone->max_vel;
    env->observations[4] = linear_vel_body.y / drone->max_vel;
    env->observations[5] = linear_vel_body.z / drone->max_vel;

    // Angular velocity
    env->observations[6] = drone->omega.x / drone->max_omega;
    env->observations[7] = drone->omega.y / drone->max_omega;
    env->observations[8] = drone->omega.z / drone->max_omega;

    // Drone orientation (up vector in world frame)
    env->observations[9] = drone_up_world.x;
    env->observations[10] = drone_up_world.y;
    env->observations[11] = drone_up_world.z;

    // Quaternion
    env->observations[12] = drone->quat.w;
    env->observations[13] = drone->quat.x;
    env->observations[14] = drone->quat.y;
    env->observations[15] = drone->quat.z;
}

void c_reset(DroneCrazyflie *env) {
    env->tick = 0;
    env->score = 0;
    env->episodic_return = 0.0f;

    env->moves_left = env->max_moves;

    Drone *drone = &env->drone;

    float size = rndf(0.05f, 0.8);
    init_drone(drone, size, 0.1f);
    
    // Start drone at a random position in the environment
    drone->pos = (Vec3){rndf(-5, 5), rndf(-5, 5), rndf(-5, 5)};

    drone->prev_pos = drone->pos;
    drone->vel = (Vec3){0.0f, 0.0f, 0.0f};
    drone->omega = (Vec3){0.0f, 0.0f, 0.0f};
    drone->quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    compute_observations(env);
}

void c_step(DroneCrazyflie *env) {

    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->log.score = 0;

    Drone *drone = &env->drone;
    move_drone(drone, env->actions);

    // Check out of bounds
    bool out_of_bounds = drone->pos.x < -GRID_SIZE || drone->pos.x > GRID_SIZE ||
                         drone->pos.y < -GRID_SIZE || drone->pos.y > GRID_SIZE ||
                         drone->pos.z < -GRID_SIZE || drone->pos.z > GRID_SIZE;

    if (out_of_bounds) {
        env->rewards[0] -= 1;
        env->episodic_return -= 1;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        compute_observations(env);
        return;
    }

    // Hovering reward: reward for being close to origin and having low velocity
    float distance_to_origin = norm3(drone->pos);
    float velocity_magnitude = norm3(drone->vel);
    
    // Reward for being close to origin (closer = higher reward)
    float distance_reward = expf(-distance_to_origin * 0.5f);
    
    // Reward for low velocity (encourages hovering)
    float velocity_reward = expf(-velocity_magnitude * 2.0f);
    
    // Combined reward
    float reward = distance_reward * velocity_reward * 0.1f;
    
    env->rewards[0] += reward;
    env->episodic_return += reward;

    // Truncate when max moves reached
    env->moves_left -= 1;
    if (env->moves_left == 0) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    drone->prev_pos = drone->pos;

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(DroneCrazyflie *env) {
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

Client *make_client(DroneCrazyflie *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    //SetConfigFlags(FLAG_WINDOW_UNDECORATED);
    InitWindow(WIDTH, HEIGHT, "PufferLib DroneRace");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 8.0f;  // Much closer (was 40.0f)
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 10.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    // Initialize trail buffer
    client->trail_index = 0;
    client->trail_count = 0;
    Drone *drone = &env->drone;
    for (int i = 0; i < TRAIL_LENGTH; i++) {
        client->trail[i] = drone->pos;
    }

    return client;
}

void c_render(DroneCrazyflie *env) {
    Drone *drone = &env->drone;
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
    client->trail[client->trail_index] = drone->pos;
    client->trail_index = (client->trail_index + 1) % TRAIL_LENGTH;
    if (client->trail_count < TRAIL_LENGTH)
        client->trail_count++;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(client->camera);

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_SIZE * 2.0f, GRID_SIZE * 2.0f, GRID_SIZE * 2.0f,
                  WHITE);

    // Draw target marker at origin
    DrawSphere((Vector3){0.0f, 0.0f, 0.0f}, 0.2f, GREEN);
    DrawSphereWires((Vector3){0.0f, 0.0f, 0.0f}, 0.5f, 8, 8, LIME);

    // draws drone body
    float r = drone->arm_len;
    float visual_scale = 2.0f;
    DrawSphere((Vector3){drone->pos.x, drone->pos.y, drone->pos.z}, r/2.0f * visual_scale, RED);

    // draws rotors according to thrust
    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = (env->actions[i] + 1.0f) * 0.5f * drone->max_rpm;
        T[i] = drone->k_thrust * rpm * rpm;
    }

    const float rotor_radius = r / 4.0f * visual_scale;  // Make rotors bigger too
    const float visual_arm_len = 1.0f * drone->arm_len;

    Vec3 rotor_offsets_body[4] = {{+r, 0.0f, 0.0f},
                                  {-r, 0.0f, 0.0f},
                                  {0.0f, +r, 0.0f},
                                  {0.0f, -r, 0.0f}};

    Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};

    for (int i = 0; i < 4; i++) {
        Vec3 world_off = quat_rotate(drone->quat, rotor_offsets_body[i]);

        Vector3 rotor_pos = {drone->pos.x + world_off.x, drone->pos.y + world_off.y,
                             drone->pos.z + world_off.z};

        float rpm = (env->actions[i] + 1.0f) * 0.5f * drone->max_rpm;
        float intensity = 0.75f + 0.25f * (rpm / drone->max_rpm);

        Color rotor_color = (Color){(unsigned char)(base_colors[i].r * intensity),
                                    (unsigned char)(base_colors[i].g * intensity),
                                    (unsigned char)(base_colors[i].b * intensity), 255};

        DrawSphere(rotor_pos, rotor_radius, rotor_color);

        DrawCylinderEx((Vector3){drone->pos.x, drone->pos.y, drone->pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                       BLACK);
    }

    // draws line with direction and magnitude of velocity / 10
    if (norm3(drone->vel) > 0.1f) {
        DrawLine3D((Vector3){drone->pos.x, drone->pos.y, drone->pos.z},
                   (Vector3){drone->pos.x + drone->vel.x * 0.1f, drone->pos.y + drone->vel.y * 0.1f,
                             drone->pos.z + drone->vel.z * 0.1f},
                   MAGENTA);
    }

    // Draw trailing path
    for (int i = 1; i < client->trail_count; i++) {
        int idx0 = (client->trail_index + i) % TRAIL_LENGTH;
        int idx1 = (client->trail_index + i - 1) % TRAIL_LENGTH;
        float alpha = (float)i / client->trail_count * 0.8f; // fade out
        Color trail_color = ColorAlpha(YELLOW, alpha);
        DrawLine3D((Vector3){client->trail[idx0].x, client->trail[idx0].y, client->trail[idx0].z},
                   (Vector3){client->trail[idx1].x, client->trail[idx1].y, client->trail[idx1].z},
                   trail_color);
    }

    EndMode3D();

    // Draw 2D stats
    DrawText(TextFormat("Distance to Origin: %.2f", norm3(drone->pos)), 10, 10, 20, WHITE);
    DrawText(TextFormat("Moves left: %d", env->moves_left), 10, 40, 20, WHITE);
    DrawText(TextFormat("Episode Return: %.2f", env->episodic_return), 10, 70, 20, WHITE);

    DrawText("Motor Thrusts:", 10, 110, 20, WHITE);
    DrawText(TextFormat("Front: %.3f", T[0]), 10, 135, 18, ORANGE);
    DrawText(TextFormat("Back:  %.3f", T[1]), 10, 155, 18, PURPLE);
    DrawText(TextFormat("Right: %.3f", T[2]), 10, 175, 18, LIME);
    DrawText(TextFormat("Left:  %.3f", T[3]), 10, 195, 18, SKYBLUE);

    DrawText(TextFormat("Pos: (%.1f, %.1f, %.1f)", drone->pos.x, drone->pos.y, drone->pos.z), 10, 225, 18,
             WHITE);
    DrawText(TextFormat("Vel: %.2f m/s", norm3(drone->vel)), 10, 245, 18, WHITE);

    DrawText("Left click + drag: Rotate camera", 10, 275, 16, LIGHTGRAY);
    DrawText("Mouse wheel: Zoom in/out", 10, 295, 16, LIGHTGRAY);

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
