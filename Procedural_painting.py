# reference ==> https://shahriyarshahrabi.medium.com/procedural-paintings-with-genetic-evolution-algorithm-6838a6e64703
import taichi
import taichi as ti
import numpy as np
from PIL import Image
import random

ti.init(arch=ti.cuda)


@ti.func
def circle(pos, center, radius, blur):
    r = (pos - center).norm()
    t = 0.0
    if blur > 1.0: blur = 1.0
    if blur <= 0.0:
        t = 1.0 - step(1.0, r / radius)
    else:
        t = smoothstep(1.0, 1.0 - blur, r / radius)
    return t


@ti.func
def step(edge, v):
    ret = 0.0
    if v < edge:
        ret = 0.0
    else:
        ret = 1.0
    return ret


@ti.func
def smoothstep(edge1, edge2, v):
    assert (edge1 != edge2)
    t = (v - edge1) / float(edge2 - edge1)
    t = clamp(t, 0.0, 1.0)

    return (3 - 2 * t) * t ** 2


@ti.func
# 0到1取原值，大于1取1
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)


@ti.func
def fract(vec):
    return vec - ti.floor(vec)


def r():
    return random.random()


# 加载初始图像
load_image = np.array(Image.open('Mona_Lisa.jpg'), dtype=np.float32)
load_image /= 255.0
ori_x = load_image.shape[1]
ori_y = load_image.shape[0]
ori_image = ti.Vector.field(3, dtype=ti.f32, shape=(ori_y, ori_x))
ori_image.from_numpy(load_image)

rotated_image = ti.Vector.field(3, dtype=ti.f32, shape=(ori_x, ori_y))

res_x = 500
res_y = int(res_x / ori_x * ori_y)
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))
loaded = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

print(load_image.shape)
print(res_x, res_y)


@ti.kernel
def rotate():
    for i, j in ori_image:
        rotated_image[j, ori_y - 1 - i] = ori_image[i, j]


@ti.kernel
def scale():
    for i, j in loaded:
        loaded[i, j] = rotated_image[i / res_x * ori_x, j / res_y * ori_y]


@taichi.kernel
def initialize():
    for i, j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])
        pixels[i, j] = color


stroke_size = ti.field(dtype=ti.f32, shape=())
stroke_size[None] = 100
randomX = ti.field(dtype=ti.f32, shape=())
randomY = ti.field(dtype=ti.f32, shape=())


@taichi.kernel
def paint(t: ti.f32):
    center = ti.Vector([ti.floor(randomX[None] * res_x), ti.floor(randomY[None] * res_y)])
    this_color = loaded[center[0], center[1]]

    for i, j in pixels:
        color = pixels[i, j]

        pos = ti.Vector([i, j])
        blur = fract(ti.sin(float(0.1 * t + i * 5 + j * 3)))
        scale = fract(ti.sin(float(0.1 * t))) + 0.5
        c = circle(pos, center, stroke_size[None] * scale, blur)

        if c > 0.9:
            color = this_color * c

        pixels[i, j] = color

    if stroke_size[None] > (res_x+res_y)/20:
        stroke_size[None] -= 0.5
    elif stroke_size[None] > (res_x+res_y)/100 and stroke_size[None] >3:
        stroke_size[None] -= 0.05


rotate()
scale()
initialize()
paused = False
speed = 10
gui = ti.GUI("screen", (res_x, res_y))
for i in range(100000):

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.SPACE:
            paused = not paused
            print("paused =", paused)

    if paused:
        for t in range(speed):
            randomX[None] = r()
            randomY[None] = r()
            paint(i)

    gui.set_image(pixels)
    gui.show()
