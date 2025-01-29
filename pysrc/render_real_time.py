# File: windowed_texture_renderer.py

import sys
import glfw
import OpenGL.GL as gl
from PIL import Image
import numpy as np  # Import NumPy
import mitsuba as mi
import drjit as dr

# Vertex and fragment shader sources
VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

out vec2 vTexCoord;

uniform mat4 transform;

void main() {
    vTexCoord = texCoord;
    gl_Position = transform * vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, vec2(vTexCoord.x , 1.0f - vTexCoord.y));
}
"""

# Camera movement variables
camera_movement = {'forward': 0, 'backward': 0, 'left': 0, 'right': 0}
movement_speed = 0.1  # Adjust movement speed here

mouse_pressed = False
last_mouse_x, last_mouse_y = 0, 0
yaw, pitch = 0.0, 0.0  # Camera rotation angles
rotation_speed = 0.005  # Adjust for sensitivity


def mouse_button_callback(window, button, action, mods):
    """GLFW mouse button callback."""
    global mouse_pressed
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_pressed = (action == glfw.PRESS)
        if not mouse_pressed:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)  # Release mouse
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)  # Capture mouse


def cursor_position_callback(window, xpos, ypos):
    """GLFW cursor position callback for rotating the camera."""
    global last_mouse_x, last_mouse_y, yaw, pitch

    if mouse_pressed:
        # Calculate mouse deltas
        dx = xpos - last_mouse_x
        dy = ypos - last_mouse_y

        # Update yaw and pitch based on mouse movement
        yaw += dx * rotation_speed
        pitch += dy * rotation_speed

        # Constrain pitch to avoid flipping the camera upside-down
        pitch = np.clip(pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

    # Store current mouse position for the next frame
    last_mouse_x, last_mouse_y = xpos, ypos


def key_callback(window, key, scancode, action, mods):
    """Handle keyboard input for camera movement."""
    global camera_movement
    if action == glfw.PRESS:
        if key == glfw.KEY_W:
            camera_movement['forward'] = 1
        elif key == glfw.KEY_S:
            camera_movement['backward'] = 1
        elif key == glfw.KEY_A:
            camera_movement['left'] = 1
        elif key == glfw.KEY_D:
            camera_movement['right'] = 1
    elif action == glfw.RELEASE:
        if key == glfw.KEY_W:
            camera_movement['forward'] = 0
        elif key == glfw.KEY_S:
            camera_movement['backward'] = 0
        elif key == glfw.KEY_A:
            camera_movement['left'] = 0
        elif key == glfw.KEY_D:
            camera_movement['right'] = 0


def update_camera(scene):
    """Update the camera's position based on user input."""
    global camera_movement, movement_speed, yaw, pitch

    # Access scene parameters
    params = mi.traverse(scene)
    # Extract current camera position and target
    transform = params['Camera-camera.to_world']
    matrix = transform.matrix
    cam_position = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    cam_target = -np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]])
    up = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]])
    # Normalize forward direction
    forward = -(cam_target / np.linalg.norm(cam_target))
    right = np.cross(forward, up, axis=0)
    right /= np.linalg.norm(right)

    # Update camera position based on user input
    move_vector = np.array([[0.0], [0.0], [0.0]])
    if camera_movement['forward']:
        move_vector += forward
    if camera_movement['backward']:
        move_vector -= forward
    if camera_movement['left']:
        move_vector -= right
    if camera_movement['right']:
        move_vector += right

    move_vector *= movement_speed

    translate = mi.Transform4f().translate([move_vector[0], move_vector[1], move_vector[2]])

    # Create a rotation matrix from yaw and pitch
    yaw_rotation = mi.Transform4f().rotate(axis=mi.Point3f(0, 0, 1), angle=dr.scalar.Float(np.degrees(yaw)))
    pitch_rotation = mi.Transform4f().rotate(axis=mi.Point3f(right[0], right[1], right[2]), angle=dr.scalar.Float(np.degrees(pitch)))
    yaw = 0.0
    pitch = 0.0
    translate_to_origin = mi.Transform4f().translate([-cam_position[0], -cam_position[1], -cam_position[2]])
    translate_back = mi.Transform4f().translate([cam_position[0], cam_position[1], cam_position[2]])
    new_transform = translate @ transform

    # Assign the new position back to the parameters
    params['Camera-camera.to_world'] = new_transform
    params.update()  # Apply changes


def compile_shader(source, shader_type):
    """Compile a shader."""
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    # Check for compilation errors
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed: {error}")

    return shader


def create_shader_program():
    """Create and link a shader program."""
    vertex_shader = compile_shader(VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(FRAGMENT_SHADER_SRC, gl.GL_FRAGMENT_SHADER)

    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    # Check for linking errors
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        error = gl.glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {error}")

    # Clean up shaders as they're no longer needed
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    return program


def calculate_aspect_ratio(image_width, image_height, window_width, window_height):
    image_aspect = image_width / image_height
    window_aspect = window_width / window_height

    if image_aspect > window_aspect:
        scale_x = window_aspect / image_aspect
        scale_y = 1.0
    else:
        scale_x = 1.0
        scale_y = image_aspect / window_aspect

    return scale_x, scale_y


def create_transformation_matrix(scale_x, scale_y):
    """Create a transformation matrix for scaling."""
    return np.array([
        [scale_x, 0, 0, 0],
        [0, scale_y, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def load_texture_from_mitsuba(scene, width, height):
    """Renders an image using Mitsuba and loads it as a texture."""
    try:
        # Load the scene
        params = mi.traverse(scene)

        # Set the film resolution
        width = params['Camera-camera.film.size'][0]
        height = params['Camera-camera.film.size'][1]

        # Render the scene
        image = mi.render(scene, spp=4)
        denoiser = mi.OptixDenoiser(input_size=[width, height], albedo=False, normals=False, temporal=False)
        denoised = denoiser(image)
        bmp = mi.Bitmap(denoised)
        mi.util.write_bitmap("output.png", image)
        print(bmp)
        bmp = bmp.convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.Float32,
            srgb_gamma=True
        )
        # Convert the image to a NumPy array of unsigned bytes
        image_data = np.array(bmp, dtype=np.float32)
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
            gl.GL_RGB, gl.GL_FLOAT, image_data
        )
        return texture, width, height
    except Exception as e:
        print(f"Error during Mitsuba render/texture load: {e}")
        return None, None, None


def main():
    scene_path = "C:/dev/LiverRenderer/resources/data/scenes/liver/liver.xml"
    scene = mi.load_file(scene_path)
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Create a windowed mode window
    window_width, window_height = 683, 512
    window = glfw.create_window(window_width, window_height, "Windowed Texture Renderer", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Set up key callback
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    # Set up OpenGL
    shader_program = create_shader_program()

    # Vertex data (unit quad)
    vertices = [
        # Positions    # Texture Coords
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        -1.0, 1.0, 0.0, 1.0
    ]
    indices = [
        0, 1, 2,
        2, 3, 0
    ]

    vertices = (gl.GLfloat * len(vertices))(*vertices)
    indices = (gl.GLuint * len(indices))(*indices)

    # Create VAO and VBO
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    ebo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices, gl.GL_STATIC_DRAW)

    # Specify vertex attributes
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * gl.sizeof(gl.GLfloat), None)
    gl.glEnableVertexAttribArray(0)

    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * gl.sizeof(gl.GLfloat),
                             gl.ctypes.c_void_p(2 * gl.sizeof(gl.GLfloat)))
    gl.glEnableVertexAttribArray(1)

    gl.glBindVertexArray(0)

    # Load the texture initially from mitsuba
    # Main loop
    while not glfw.window_should_close(window):
        update_camera(scene)
        texture, image_width, image_height = load_texture_from_mitsuba(scene, window_width, window_height)
        if texture is None:
            glfw.terminate()
            return
        # Get current window size
        window_width, window_height = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, window_width, window_height)

        # Calculate scale to maintain aspect ratio
        scale_x, scale_y = calculate_aspect_ratio(image_width, image_height, window_width, window_height)

        # Create the transformation matrix
        transform_matrix = create_transformation_matrix(scale_x, scale_y)

        # Clear screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Use the shader program and bind the texture
        gl.glUseProgram(shader_program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        # Set transformation matrix for aspect ratio
        transform_location = gl.glGetUniformLocation(shader_program, "transform")
        gl.glUniformMatrix4fv(transform_location, 1, gl.GL_FALSE, transform_matrix)

        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()
        gl.glDeleteTextures([texture])

    # Cleanup
    gl.glDeleteTextures([texture])
    gl.glDeleteVertexArrays(1, [vao])
    gl.glDeleteBuffers(1, [vbo, ebo])
    glfw.terminate()


if __name__ == "__main__":
    import mitsuba as mi

    mi.set_variant('cuda_ad_spectral')

    main()
