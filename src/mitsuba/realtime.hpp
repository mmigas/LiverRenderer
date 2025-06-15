// --- Includes ---
// (Keep necessary Mitsuba, Dr.Jit includes)
#include <mitsuba/core/logger.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>

#include <drjit-core/jit.h>
#include <drjit/autodiff.h>
#include <drjit/matrix.h>
#include <drjit/tensor.h>
#include <mitsuba/render/optixdenoiser.h>
// Add GL/GLFW includes back here
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Add CUDA includes ONLY if still needed by Dr.Jit/Mitsuba backend itself
// Remove <cudaGL.h> as interop is gone
#include <cuda_runtime.h> // Still needed for cudaStreamSynchronize maybe? Or use dr::sync_thread()

#include <cmath>
#include <corecrt_math_defines.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
// Maybe #include <omp.h> if using OpenMP pragma for copy loop

// -----------------------------------------------------------------------------

namespace mitsuba {

// --- REMOVE CUDA Error Checking Macro if not needed ---
// #define CHECK_CUDA(call) ...

// --- Camera Struct Definition --- (Use the last version with Delta Rotation + Roll)
struct Camera {
    // ... (Exact same definition as the last version provided) ...
    Point<float, 3> Position = { 0.0f, 0.0f, 3.0f };
    Vector<float, 3> Front   = { 0.0f, 0.0f, -1.0f };
    Vector<float, 3> Up      = { 0.0f, 1.0f, 0.0f };
    Vector<float, 3> Right   = { 1.0f, 0.0f, 0.0f };

    // Orientation
    float Yaw   = -90.0f;
    float Pitch = 0.0f;

    // Settings
    float MovementSpeed    = 7.5f;
    float MouseSensitivity = 0.1f;
    float RollSpeed        = 100.5f; // Radians per second for roll <-- ADD THIS

    // Mouse Tracking
    bool firstMouse = true;
    float lastX     = 0.0f;
    float lastY     = 0.0f;

    bool updated = false;
    Camera()     = default;

    static float radians(float degrees) { return degrees * (M_PI / 180.0f); }
    static float degrees(float radians) { return radians * (180.0f / M_PI); }

    // --- Initialization ---
    void initializeFromTransform(const Transform<Point<float, 4>> &transform) {
        // Use the corrected vector extraction method
        Log(Info, "Initializing Camera Vectors (Corrected v2):\n%s", transform);
        Position = transform * mitsuba::Point<float, 3>(0.f, 0.f, 0.f);
        Right    = dr::normalize(transform * mitsuba::Vector<float, 3>(-1.f, 0.f, 0.f)); // World X = Local X
        Up       = dr::normalize(transform * mitsuba::Vector<float, 3>(0.f, 1.f, 0.f));  // World Y = Local Y
        // Local forward is -Z in typical camera space. Transform it to get world forward.
        Front = dr::normalize(transform * mitsuba::Vector<float, 3>(0.f, 0.f, 1.f)); // World Front =
        // Local -Z
        if (std::abs(Front.y()) < 0.9999f) {
            Yaw = degrees(std::atan2(Front.z(), Front.x()));
        } else { // Looking straight up/down
            // Use Right vector's XZ projection for Yaw
            Yaw = degrees(std::atan2(Right.z(), Right.x()));
        }
        float front_y_clamped = dr::clamp(Front.y(), -1.0f, 1.0f);
        Pitch                 = degrees(std::asin(front_y_clamped));
        Log(Info, " Final Initial Yaw=%.2f, Pitch=%.2f", Yaw, Pitch);
        // Reset mouse tracking for the first input event
        firstMouse = true;
    }

    // --- Input Processing Methods ---
    void processKeyboard(GLFWwindow *window, float deltaTime) {
        // Version with WASD translation + QE roll
        float moveVelocity        = MovementSpeed * deltaTime;
        float rollAngleDelta      = RollSpeed * deltaTime;
        Vector<float, 3> deltaPos = { 0.0f, 0.0f, 0.0f };
        float currentFrameRoll    = 0.0f;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            deltaPos += Front * moveVelocity;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            deltaPos -= Front * moveVelocity;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            deltaPos -= Right * moveVelocity;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            deltaPos += Right * moveVelocity;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            deltaPos += Up * moveVelocity;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            deltaPos -= Up * moveVelocity; // Use own Up

        Position += deltaPos;

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            currentFrameRoll += rollAngleDelta;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            currentFrameRoll -= rollAngleDelta;

        if (std::abs(currentFrameRoll) > 1e-7f) {
            Transform<Point<float, 4>> rotRoll = Transform<Point<float, 4>>::rotate(Front, currentFrameRoll);
            Up                                 = dr::normalize(rotRoll * Up);
            Right                              = dr::normalize(rotRoll * Right);
            // Re-orthogonalize
            Right = dr::normalize(dr::cross(Front, Up));
            Up    = dr::normalize(dr::cross(Right, Front));
        }
        if (deltaPos.x() != 0.0f || deltaPos.y() != 0.0f || deltaPos.z() != 0.0f || currentFrameRoll != 0.0f) {
            updated = true;
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

    void processMouseMovement(float xoffset, float yoffset) {
        // Delta rotation implementation
        float yawAngle   = -xoffset * MouseSensitivity;
        float pitchAngle = -yoffset * MouseSensitivity;

        // Pitch clamping (approximate)
        Transform<Point<float, 4>> rotPitchCheck = Transform<Point<float, 4>>::rotate(Right, pitchAngle);
        Vector<float, 3> potentialFront          = rotPitchCheck * Front;
        constexpr float pitchLimitY              = 0.999f; // ~87 deg
        if (potentialFront.y() > pitchLimitY || potentialFront.y() < -pitchLimitY) {
            pitchAngle = 0.0f;
        }

        // Apply Yaw (around Up)
        if (std::abs(yawAngle) > 1e-7f) {
            Transform<Point<float, 4>> rotYaw = Transform<Point<float, 4>>::rotate(Up, yawAngle);
            Front                             = rotYaw * Front;
            Up                                = rotYaw * Up;
            Right                             = rotYaw * Right;
        }

        // Apply Pitch (around current Right)
        if (std::abs(pitchAngle) > 1e-7f) {
            Transform<Point<float, 4>> rotPitch = Transform<Point<float, 4>>::rotate(-Right, pitchAngle);
            Front                               = rotPitch * Front;
            Up                                  = rotPitch * Up;
        }

        // Re-normalize & Re-orthogonalize (important!)
        if (std::abs(yawAngle) > 1e-7f || std::abs(pitchAngle) > 1e-7f) {
            Front = dr::normalize(Front);
            // Use Up to keep Right relatively horizontal after yaw/pitch adjustments
            Right = dr::normalize(dr::cross(Front, Up));
            Up    = dr::normalize(dr::cross(Right, Front));
        }

        if (yawAngle != 0.0f || pitchAngle != 0.0f) {
            updated = true;
        }
    }
};

// --- GLFW Callback Function --- (Same as before)
inline void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    // Retrieve the pointer stored during initialization
    Camera *camera = static_cast<Camera *>(glfwGetWindowUserPointer(window));
    // Check if the pointer is valid (it should be if set correctly)
    if (!camera) {
        return;
    }

    // Handle first mouse movement
    if (camera->firstMouse) {
        camera->lastX      = (float) xpos;
        camera->lastY      = (float) ypos;
        camera->firstMouse = false;
    }

    // Calculate offset from last frame
    float xoffset = (float) xpos - camera->lastX;
    float yoffset = camera->lastY - (float) ypos; // Reversed y-coordinates

    // Update last mouse position
    camera->lastX = (float) xpos;
    camera->lastY = (float) ypos;

    // Apply sensitivity and process movement
    camera->processMouseMovement(xoffset, yoffset);
}

// --- *** Add GLFW/GLEW Init Helpers Back Here *** ---
static GLFWwindow *init_glfw(int width, int height, const std::string &title) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    Log(Info, "GLFW Initialized.");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    GLFWwindow *window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    Log(Info, "GLFW Window Created.");
    return window;
}

static void init_glew() {
    glewExperimental = GL_TRUE;
    GLenum err       = glewInit();
    if (GLEW_OK != err) {
        std::stringstream ss;
        ss << "Error initializing GLEW: " << glewGetErrorString(err);
        glfwTerminate(); // Terminate GLFW if GLEW fails
        throw std::runtime_error(ss.str());
    }
    while (glGetError() != GL_NO_ERROR)
        ; // Clear potential GLEW init error
    Log(Info, "GLEW Initialized. Version: %s", (const char *) glewGetString(GLEW_VERSION));
}
// --- End GLFW/GLEW Init Helpers ---

// --- OpenGL Helper Functions --- (Same as before)
static GLuint compile_shader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::string shaderType = (type == GL_VERTEX_SHADER) ? "VERTEX" : "FRAGMENT";
        std::stringstream ss;
        ss << "ERROR::SHADER::" << shaderType << "::COMPILATION_FAILED\n" << infoLog;
        glDeleteShader(shader);
        throw std::runtime_error(ss.str());
    }
    return shader;
}

GLuint link_shader_program(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::stringstream ss;
        ss << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog;
        glDeleteProgram(program);
        throw std::runtime_error(ss.str());
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

// Setup quad buffers (same as before)
void setup_quad_buffers(GLuint &vao, GLuint &vbo) {
    float vertices[] = { -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Create GL texture (same as before, assuming RGBA32F)
GLuint create_gl_texture(int width, int height) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Still assuming RGBA32F for direct float copy from CUDA
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}
// --- Shader Sources --- (Same as before)
const char *vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y); // Flip vertically
}
)glsl";

const char *fragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)glsl";

enum class InteractiveMode { EMA, Optix, Unknown };

// --- Main Real-time Render Function Implementation ---
template <typename Float, typename Spectrum> void runRealtimeRenderer(Object *scene_obj, const std::string &interactive_mode_str) {
    MI_IMPORT_CORE_TYPES()
    InteractiveMode current_mode = InteractiveMode::Unknown;
    if (interactive_mode_str == "ema") {
        current_mode = InteractiveMode::EMA;
    } else if (interactive_mode_str == "optix") {
        current_mode = InteractiveMode::Optix;
    }

    Log(Info, "OpenGL context initialized successfully.");
    auto *scene = dynamic_cast<Scene<Float, Spectrum> *>(scene_obj);
    if (!scene) {
        throw std::runtime_error("Invalid scene object passed.");
    }

    Sensor<Float, Spectrum> *sensor         = scene->sensors().empty() ? nullptr : scene->sensors()[0];
    Integrator<Float, Spectrum> *integrator = scene->integrator();
    Film<Float, Spectrum> *film             = sensor ? sensor->film() : nullptr;
    if (!sensor || !integrator) {
        throw std::runtime_error("Scene lacks sensor, integrator, or film.");
    }

    ScalarVector2u film_size = film->size();
    int width                = film_size.x();
    int height               = film_size.y();
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Invalid film dimensions.");

    // --- Local variables for GL/GLFW ---
    GLFWwindow *window = nullptr;
    GLuint glTextureID = 0, quadVAO = 0, quadVBO = 0, shaderProgram = 0;
    std::vector<float> cpu_texture_buffer; // For CPU transfer path

    Camera camera; // Local instance

    // --- State for Accumulation (EMA)
    TensorXf current_average_gpu;
    bool average_initialized = false;
    const float EMA_ALPHA    = 0.01f;

    // --- State for OptiX Denoiser
    // Use the Mitsuba wrapper class, templated on Float/Spectrum
    std::unique_ptr<OptixDenoiser<Float, Spectrum>> denoiser;
    TensorXf denoised_buffer_gpu; // To store denoiser output

    // --- State for Change Detection
    mitsuba::Timer stageTimer; // Single timer, reset for each stage
    double totalInputTimeMs   = 0.0;
    double totalRenderTimeMs  = 0.0;
    double totalAccumTimeMs   = 0.0;
    double totalDisplayTimeMs = 0.0; // Includes migrate, CPU copy, GL upload, draw, swap
    uint64_t totalFrameCount  = 0;
    double totalAppStartTime  = glfwGetTime(); // Record start time
    try {
        // --- Initialize GLFW/GLEW HERE ---
        Log(Info, "Initializing GLFW/GLEW...");
        window = init_glfw(width, height, "Mitsuba 3 Realtime Renderer");
        init_glew();                     // Requires active context from init_glfw
        glViewport(0, 0, width, height); // Set viewport after init
        glEnable(GL_FRAMEBUFFER_SRGB);

        // --- Setup Camera (AFTER GL/GLFW init if needed for mouse pos) ---
        ScalarTransform4f initial_transform_scalar(sensor->m_to_world.scalar());
        camera.initializeFromTransform(initial_transform_scalar);
        glfwSetCursorPos(window, (double) width / 2.0, (double) height / 2.0); // Center cursor
        camera.lastX      = float(width) / 2.0f;
        camera.lastY      = float(height) / 2.0f;
        camera.firstMouse = true;

        // --- Setup GLFW Callbacks ---
        glfwSetWindowUserPointer(window, &camera);
        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // --- OpenGL Resources ---
        Log(Info, "Setting up OpenGL resources...");
        glTextureID = create_gl_texture(width, height);
        setup_quad_buffers(quadVAO, quadVBO);
        GLuint vertShader = compile_shader(GL_VERTEX_SHADER, vertexShaderSource);
        GLuint fragShader = compile_shader(GL_FRAGMENT_SHADER, fragmentShaderSource);
        shaderProgram     = link_shader_program(vertShader, fragShader);
        glUseProgram(shaderProgram);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);
        glUseProgram(0); // Unbind shader initially

        // --- Initialize CPU buffer for texture data ---
        cpu_texture_buffer.resize((size_t) width * height * 4); // RGBA

        if (current_mode == InteractiveMode::Optix) {
            Log(Info, "[Realtime] Initializing OptiX Denoiser...");
            try {
                denoiser = std::make_unique<OptixDenoiser<Float, Spectrum>>(ScalarVector2u(width, height),
                                                                            false, // useAlbedo - TODO: Make configurable later?
                                                                            false, // useNormals - TODO: Make configurable later?
                                                                            false  // useTemporal
                );
                // Pre-allocate output buffer if denoiser doesn't manage it
                // Check OptixDenoiser API - assuming it returns a new tensor for now
                Log(Info, "[Realtime] OptiX Denoiser initialized.");
            } catch (const std::exception &e) {
                Log(Error, "[Realtime] Failed to initialize OptiX Denoiser: %s. Falling back to EMA.", e.what());
                // Fallback strategy: Switch mode to EMA
                current_mode = InteractiveMode::EMA;
                denoiser     = nullptr; // Ensure pointer is null
            }
        }

        // --- Render Loop ---
        Log(Info, "Starting render loop...");
        double lastFrameTime        = glfwGetTime();
        int frameCount              = 0;
        double fpsLastTime          = lastFrameTime;
        std::string baseWindowTitle = "Mitsuba 3 Realtime";

        while (!glfwWindowShouldClose(window)) {
            double frameStartTime = glfwGetTime();
            float deltaTime       = (float) (frameStartTime - lastFrameTime);
            lastFrameTime         = frameStartTime;
            frameCount++;
            totalFrameCount++;

            // --- FPS Counter --- (Same as before)
            if (frameStartTime - fpsLastTime >= 1.0) {
                double fps = double(frameCount) / (frameStartTime - fpsLastTime);
                std::stringstream ss;
                ss << baseWindowTitle << " [" << static_cast<int>(fps) << " FPS]";
                glfwSetWindowTitle(window, ss.str().c_str());

                // Reset counter
                frameCount  = 0;
                fpsLastTime = frameStartTime;
            }

            // --- Process Input --- (Same as before)
            stageTimer.reset();

            glfwPollEvents();
            camera.processKeyboard(window, deltaTime);
            Transform4f new_to_world_scalar = Transform4f::look_at(camera.Position,                // Camera position
                                                                   camera.Position + camera.Front, // Where the camera looks
                                                                   camera.Up                       // Use fixed world up ({0,1,0}) for orientation
            );

            // Construct the JIT Transform4f (using the local alias) from the
            // scalar matrix
           sensor->m_to_world = Transform4f(new_to_world_scalar.matrix);
           sensor->parameters_changed({ "to_world" }); // Notify Mitsuba

            totalInputTimeMs += stageTimer.value(); // Convert to ms

            // --- Render Current Frame --- (Same as before)
            stageTimer.reset();
            uint32_t current_seed      = (uint32_t) (totalFrameCount & 0xFFFFFFFF);
            TensorXf current_frame_gpu = integrator->render(scene, sensor, current_seed);
            dr::sync_thread();
            totalRenderTimeMs += stageTimer.value();
            // Use dr::sync_thread() here IF migrate needs it, maybe not needed before EMA

            // --- Update Exponential Moving Average --- (Same as before)
            stageTimer.reset();
            if (current_mode == InteractiveMode::Optix) {
                // Use OptiX denoiser for accumulation
                if (denoiser) {
                    current_average_gpu = (*denoiser)(current_frame_gpu);
                } else {
                    Log(Error, "OptiX Denoiser not initialized, falling back to EMA.");
                    current_mode = InteractiveMode::EMA;
                }
            } else if (current_mode == InteractiveMode::EMA) {
                if (!average_initialized || camera.updated) { /* ... init average ... */
                    average_initialized = true;
                    current_average_gpu = current_frame_gpu;
                } else { /* ... EMA calculation ... */
                    current_average_gpu = EMA_ALPHA * current_frame_gpu + (1.0f - EMA_ALPHA) * current_average_gpu;
                }
            } else {
                Log(Error, "Unknown interactive mode: %s", interactive_mode_str);
                continue; // Skip frame
            }
            // Sync may be needed before migration
            dr::sync_thread();
            totalAccumTimeMs += stageTimer.value();

            // --- *** Start: Display Averaged Result via CPU Transfer *** ---
            stageTimer.reset();
            // 1. Migrate averaged result from GPU to CPU

            auto &&result_cpu_array = dr::migrate(current_average_gpu.array(), AllocType::Host);

            if (!result_cpu_array.data()) {
                Log(Error, "Failed to get CPU data pointer from migrated tensor!");
                continue; // Skip frame
            }

            // 3. Upload CPU buffer data to OpenGL texture
            glBindTexture(GL_TEXTURE_2D, glTextureID);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, result_cpu_array.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            // --- *** End: Display Averaged Result via CPU Transfer *** ---

            // --- Render OpenGL Quad --- (Same as before)
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(shaderProgram);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, glTextureID);
            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glUseProgram(0);

            // --- Swap Buffers --- (Same as before)
            glfwSwapBuffers(window);
            totalDisplayTimeMs += stageTimer.value();
            camera.updated = false; // Reset updated flag for next frame

        } // End render loop
        double totalAppEndTime = glfwGetTime();
        double totalRunTimeSec = totalAppEndTime - totalAppStartTime;

        Log(Info, "-------------------------------");
        Log(Info, "--- Realtime Session Summary ---");
        Log(Info, "Total Runtime: %.3f s", totalRunTimeSec);
        Log(Info, "Total Frames Measured: %llu", totalFrameCount);

        if (totalFrameCount > 0 && totalRunTimeSec > 0) {
            double avgFps = (double) totalFrameCount / totalRunTimeSec;
            Log(Info, "Average FPS: %.2f", avgFps);

            double avgInputMs   = totalInputTimeMs / totalFrameCount;
            double avgRenderMs  = totalRenderTimeMs / totalFrameCount;
            double avgAccumMs   = totalAccumTimeMs / totalFrameCount;
            double avgDisplayMs = totalDisplayTimeMs / totalFrameCount;
            double avgFrameMs   = avgInputMs + avgRenderMs + avgAccumMs + avgDisplayMs;

            Log(Info, "Average Stage Timings (ms):");
            Log(Info, "  Input Handling     : %8.3f ms", avgInputMs);
            Log(Info, "  Core Ray Tracing   : %8.3f ms", avgRenderMs);
            Log(Info, "  Denoise            : %8.3f ms", avgAccumMs);
            Log(Info, "  Display            : %8.3f ms", avgDisplayMs);
            Log(Info, "  ---------------------------------");
            Log(Info, "  Avg Measured Total : %8.3f ms", avgFrameMs);
            // Note: Avg Measured Total * Avg FPS might not exactly equal 1000 due to
            // small measurement gaps or overheads not captured between stages.
        } else {
            Log(Info, "No frames measured or zero runtime, cannot calculate averages.");
        }
        Log(Info, "-------------------------------");
        // --- Cleanup inside runRealtimeRenderer ---
        Log(Info, "Cleaning up resources...");
        // --- Remove CUDA Resource Unregistration ---
        // if (g_rt_cudaResource) { cuGraphicsUnregisterResource(g_rt_cudaResource); g_rt_cudaResource = nullptr; }

        // Delete GL objects
        if (shaderProgram)
            glDeleteProgram(shaderProgram);
        if (quadVAO)
            glDeleteVertexArrays(1, &quadVAO);
        if (quadVBO)
            glDeleteBuffers(1, &quadVBO);
        if (glTextureID)
            glDeleteTextures(1, &glTextureID);

        // Restore cursor
        if (window) {

            // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }

        // --- Add GLFW Cleanup HERE ---
        if (window)
            glfwDestroyWindow(window);
        glfwTerminate(); // Terminate GLFW when this function exits
        Log(Info, "GLFW terminated.");

    } catch (const std::exception &e) {
        Log(Error, "Fatal Error: %s", e.what());
        // Ensure cleanup if error happens after GL/GLFW init
        if (window) {
            // Maybe restore cursor before destroying?
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwDestroyWindow(window);
        }
        glfwTerminate(); // Attempt termination even on error
    }
    Log(Info, "Render function finished.");
}

} // namespace mitsuba