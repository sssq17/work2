import taichi as ti
import math

# 初始化 Taichi，指定使用 CPU 后端
ti.init(arch=ti.cpu)

# 立方体：8个顶点，中心原点(0,0,0)，边长2，坐标范围[-1,1]
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 立方体的12条边（每条边两个顶点索引）
cube_edges = ti.Vector.field(2, dtype=ti.i32, shape=12)

@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32):
    """
    模型变换矩阵：X + Y + Z 三轴旋转
    """
    # X轴旋转 (W/S)
    rad_x = angle_x * math.pi / 180.0
    cx = ti.cos(rad_x)
    sx = ti.sin(rad_x)
    rot_x = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Y轴旋转 (Q/E)
    rad_y = angle_y * math.pi / 180.0
    cy = ti.cos(rad_y)
    sy = ti.sin(rad_y)
    rot_y = ti.Matrix([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Z轴旋转 (A/D)
    rad_z = angle_z * math.pi / 180.0
    cz = ti.cos(rad_z)
    sz = ti.sin(rad_z)
    rot_z = ti.Matrix([
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 组合旋转
    return rot_z @ rot_y @ rot_x

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle_x, angle_y, angle_z)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = proj @ view @ model

    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化立方体顶点
    vertices[0] = [-1.0, -1.0, -1.0]
    vertices[1] = [1.0, -1.0, -1.0]
    vertices[2] = [1.0, 1.0, -1.0]
    vertices[3] = [-1.0, 1.0, -1.0]
    vertices[4] = [-1.0, -1.0, 1.0]
    vertices[5] = [1.0, -1.0, 1.0]
    vertices[6] = [1.0, 1.0, 1.0]
    vertices[7] = [-1.0, 1.0, 1.0]

    # 12条边
    cube_edges[0] = [0, 1]
    cube_edges[1] = [1, 2]
    cube_edges[2] = [2, 3]
    cube_edges[3] = [3, 0]
    cube_edges[4] = [4, 5]
    cube_edges[5] = [5, 6]
    cube_edges[6] = [6, 7]
    cube_edges[7] = [7, 4]
    cube_edges[8] = [0, 4]
    cube_edges[9] = [1, 5]
    cube_edges[10] = [2, 6]
    cube_edges[11] = [3, 7]

    gui = ti.GUI("3D 立方体 (W/S/A/D/Q/E 旋转)", res=(700, 700))
    angle_x = 0.0
    angle_y = 0.0
    angle_z = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            # W/S → X轴
            if gui.event.key == 'w': angle_x += 10.0
            elif gui.event.key == 's': angle_x -= 10.0
            # Q/E → Y轴
            if gui.event.key == 'q': angle_y += 10.0
            elif gui.event.key == 'e': angle_y -= 10.0
            # A/D → Z轴
            if gui.event.key == 'a': angle_z += 10.0
            elif gui.event.key == 'd': angle_z -= 10.0
            # Esc 退出
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        compute_transform(angle_x, angle_y, angle_z)

        colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF]
        for i in range(12):
            a, b = cube_edges[i]
            gui.line(screen_coords[a], screen_coords[b], radius=2, color=colors[i % 6])

        gui.show()

if __name__ == '__main__':
    main()