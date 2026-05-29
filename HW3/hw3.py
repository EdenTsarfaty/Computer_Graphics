from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            hit = ray.nearest_intersected_object(objects)
            

            # This is the main loop where each pixel color is computed.
            color = get_color(ambient, lights, objects, ray, hit, max_depth)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    # light_a = SpotLight(intensity= np.array([0.788, 0.286, 0.722]), position=np.array([-0.85, 0.85, 0]), direction=([0.2, -0.85, -1.5]), kc=0.)
    pink_spot = SpotLight(
        intensity=np.array([1.0, 0.3, 0.7]), # Vibrant Pink
        position=np.array([-1.5, 1.5, 1.0]),
        direction=np.array([1.5, -1.5, -2.5]), 
        kc=0.1, kl=0.05, kq=0.01
    )

    blue_dir = DirectionalLight(
        intensity=np.array([0.2, 0.5, 0.9]), # Deep Blue
        direction=np.array([-1.0, -1.0, -1.0])
    )

    lights = [pink_spot, blue_dir]

    floor = Plane(normal=[0, 1, 0], point=[0, -1.0, 0])
    floor.set_material([0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.3, 0.3, 0.3], 10, 0.3)
    floor.refraction = 0.0

    # White Floor Plane
    floor = Plane(normal=[0, 1, 0], point=[0, -1.0, 0])
    floor.set_material([0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.3, 0.3, 0.3], 10, 0.3)
    floor.refraction = 0.0

    # Back Wall Plane
    back_wall = Plane(normal=[0, 0, 1], point=[0, 0, -4.0])
    back_wall.set_material([0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0], 1, 0.0)
    back_wall.refraction = 0.0

    # C) Transparent Diamond
    center = np.array([0.3, 0.1, -1.8])
    
    v_list = [
        np.array([-0.264, -0.228, -1.370]),
        np.array([ 0.864,  0.169, -1.264]),
        np.array([ 0.300,  0.281, -2.476]),
        np.array([ 0.026,  0.826, -1.605]),
        np.array([ 0.574, -0.626, -1.995])
    ]

    diamond = Diamond(v_list)
    diamond.set_material([0.05, 0.05, 0.05], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0], 100, 0.2)
    diamond.refraction = 0.65
    diamond.apply_materials_to_triangles() 
    
    for tri in diamond.triangle_list:
        tri.refraction = 0.65

    # Sphere Inside the Diamond
    inner_sphere = Sphere(center=center, radius=0.25)
    inner_sphere.set_material([0.2, 0.2, 0.0], [0.8, 0.7, 0.1], [0.9, 0.9, 0.9], 50, 0.1)
    inner_sphere.refraction = 0.0
    
    objects = [floor, back_wall, diamond, inner_sphere]

    return camera, lights, objects
