Here’s a concise Markdown “system spec” for your Crazyflie 2.1 MuJoCo model (without any XML):

---

## Crazyflie 2.1 (“cf2”) MuJoCo Model Specification

### 1. Model Metadata

* **Name:** cf2

---

### 2. Inertial Properties

| Property              | Value                |
| :-------------------- | :------------------- |
| Mass                  | 0.027 kg             |
| Center of mass offset | (0.0, 0.0, 0.0) m    |
| Inertia (diag.)       | Ix=2.3951×10⁻⁵ kg·m² |

```
                      Iy=2.3951×10⁻⁵ kg·m²  
                      Iz=3.2347×10⁻⁵ kg·m²  |
```

---

### 3. Geometry

* **Collision shape:** Sphere

* **Radius:** 0.055 m


### 4. Body & Sites

* **Base body “drone”** positioned at (0, 0, 0.05) m above world origin
* **Sites** (all relative to body frame):

  | Site   | Position (x y z) \[m]    | Description         |
  | :----- | :----------------------- | :------------------ |
  | CoM    | —                        | Center-of-mass viz. |
  | motor0 | ( 0.0325, –0.0325, 0.0 ) | Rotor 0 location    |
  | motor1 | (–0.0325, –0.0325, 0.0 ) | Rotor 1 location    |
  | motor2 | (–0.0325,  0.0325, 0.0 ) | Rotor 2 location    |
  | motor3 | ( 0.0325,  0.0325, 0.0 ) | Rotor 3 location    |

---

### 5. Actuators

**Per-motor force & torque channels** (4 motors: 0…3)

| Actuator type | Gear vector                | Control →         | Max output   |
| :------------ | :------------------------- | :---------------- | :----------- |
| Force         | \[0, 0, **0.32**, 0, 0, 0] | → thrust (N)      | 0.32 N/motor |
| Torque        | \[0, 0, 0, 0, 0, **1**]    | → reaction torque | 1 (unitless) |

> **Note:** 0.32 N per motor corresponds to \~32.6 g lift (≈130 g total thrust).

---