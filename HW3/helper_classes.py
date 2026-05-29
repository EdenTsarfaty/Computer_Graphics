import numpy as np

# Avoid float point computation issues
EPSILON = 1e-6
OFFSET = 1e-4


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def get_color(ambient_coe: np.ndarray, lights: list, objects: list, ray: Ray,
               hit: tuple[Object3D, float], max_level: int, level=1) -> np.ndarray:
    obj, distance = hit
    if obj is None or distance is np.inf:
        return np.array([0.0, 0.0, 0.0])
    
    hit_pt = ray.get_hit_point(distance)
    normal = obj.get_normal(hit_pt)
    # Offsetting the hit_pt along the normal by a small fraction to avoid issues
    hit_pt_offseted = hit_pt + (obj.get_normal(hit_pt) * OFFSET)

    i_emitted = np.zeros(3) # no emitted light for an object in this assignment
    phong = np.zeros(3) + i_emitted

    ambient = calc_ambient(ambient_coe, obj)
    phong += ambient

    for light in lights:
        shadow_ray = light.get_light_ray(hit_pt) # Ray from the light to the hit point
        self_dist = light.get_distance_from_light(hit_pt)

        _, occ_obj_distance = shadow_ray.nearest_intersected_object(objects)
        if occ_obj_distance < self_dist - OFFSET:
            #s_j = 0 - occluded
            continue
        
        intensity = light.get_intensity(hit_pt) #I_L
        diffuse = calc_diffuse(obj, intensity, shadow_ray, hit_pt)
        
        specular = calc_specular(obj, intensity, ray, shadow_ray, hit_pt)

        phong += diffuse + specular

    if level < max_level:
        level += 1
        reflected_ray = Ray(hit_pt_offseted, reflected(ray.direction, obj.get_normal(hit_pt)))
        reflected_obj, reflected_dist = reflected_ray.nearest_intersected_object(objects)
        if reflected_obj is not None:
            phong += obj.reflection * get_color(ambient_coe, lights, objects, reflected_ray, (reflected_obj, reflected_dist), max_level, level)

        #For scene 6
        if hasattr(obj,"refraction"):
            refraction_coe = obj.refraction
            refracted_origin = hit_pt + (ray.direction * OFFSET)

            refracted_ray = Ray(refracted_origin, ray.direction)
            refracted_obj, refaccted_dist = refracted_ray.nearest_intersected_object(objects)
            if refracted_obj is not None:
                phong += refraction_coe * get_color(ambient_coe, lights, objects, refracted_ray, (refracted_obj, refaccted_dist), max_level, level)


    return phong


def calc_ambient(ambient_coe, obj):
    return obj.ambient * ambient_coe

def calc_diffuse(obj, light_intensity, ray, hit_pt):
    diffuse_coe = obj.diffuse #K_D
    # Note that the inner product will turn out negative using a ray that goes to the intersection point
    # and a normal that goes outward from that point.
    # We therefore flip the shadow ray vector.
    # We also make sure that we get a non-negative color in case the light hits from behind the object
    light_angle_coe = max(0.0, np.inner(obj.get_normal(hit_pt), -ray.direction))
    return diffuse_coe * light_intensity * light_angle_coe

def calc_specular(obj, light_intensity, ray, shadow_ray, hit_pt):
    specular_coe = obj.specular #K_S
    shininess_f = obj.shininess #n
    ref_ray = reflected(shadow_ray.direction, obj.get_normal(hit_pt))

    # Note that the inner product will turn out negative using a ray that goes to the intersection point
    # and a normal that goes outward from that point.
    # We therefore flip the shadow ray vector.
    # We also make sure that we get a non-negative color in case the light hits from behind the object
    view_angle_coe = max(0.0, np.inner(-ray.direction, ref_ray))

    return specular_coe * light_intensity * (view_angle_coe ** shininess_f)
           
            

## Lights
class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):
    DIST_SCALAR = 100000 # Used to simulate a very far away light source from scene - increase if rendering seems incorrect

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(np.array(direction))

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point - self.direction * DirectionalLight.DIST_SCALAR, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return DirectionalLight.DIST_SCALAR
    
    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc #constant
        self.kl = kl #linear
        self.kq = kq #quadric

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(self.position, normalize(intersection - self.position))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # calculate distance between light source and intersection
        distance = self.get_distance_from_light(intersection)
        # calculate and return the light intensity based on kc, kl, kq
        attenuation = self.kq * distance**2 + self.kl * distance + self.kc

        return(self.intensity * (1 / attenuation))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.direction = normalize(np.array(direction))
        self.position = np.array(position)
        self.kc = kc #constant
        self.kl = kl #linear
        self.kq = kq #quadric

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(self.position, normalize(intersection - self.position))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        light_ray_vector = self.get_light_ray(intersection).direction
        distance = self.get_distance_from_light(intersection)
        attenuation = self.kq * distance**2 + self.kl * distance + self.kc

        # We make sure that we get a non-negative light in case the object is behind the light
        direction_factor = max(0.0, np.dot(light_ray_vector, self.direction))

        return self.intensity * (direction_factor / attenuation)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            hit = obj.intersect(self)
            if hit is not None:
                distance, obj_hit = hit
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = obj_hit

        return nearest_object, min_distance
    
    def get_hit_point (self, t):
        # get the distance in 't' value, for use in parametric representation
        # returns the point reached at distance t.
        return self.origin + t * self.direction


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + EPSILON)
        if t > EPSILON:
            return t, self
        else:
            return None
        
    def get_normal(self, hit_pt):
        return self.normal


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The front face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        vec_ab = self.b - self.a
        vec_ac = self.c - self.a
        return normalize(np.cross(vec_ab, vec_ac))

    def intersect(self, ray: Ray):
        # assert wheter the ray hits the plane containing the triangle, arbitrarily choose point ad
        triangle_plane = Plane(self.normal, self.a)
        intersection = triangle_plane.intersect(ray)
        if intersection is None:
            return None
        t, _ = triangle_plane.intersect(ray)
        if t <= 0:
            return None
        
        # assert whether the ray hit point is within the triangle using Barycentric Coordinates
        p = ray.get_hit_point(t)
        vec_ab = self.b - self.a
        vec_ac = self.c - self.a

        # Explanation: The Barycentric Coordinates method in the recitation slides (slide 19) did not work properly:
        # - Using a norm indeed returns the subtriangle area (or the parallelogram), but loses the sign.
        # - The sign is crucial for points that fall outside our triangle. Since the order of the vectors change (demonstrated by the RHR rule),
        #   the cross product has a negative result.
        # - By flipping the sign for these alpha/beta/gamma we lose their true relative proportion.
        # - We fix this by using the dot product, which results in the area, but retains the sign (since it can only go 0° or 180°, with the normal which translates to 1 or -1 with the cos function).
        # - Additionally we omit division by 2, since it cancels out
        ABC_area = np.dot(self.normal, np.cross(vec_ab, vec_ac))
        vec_pb = self.b - p
        vec_pc = self.c - p
        vec_pa = self.a - p
        
        alpha = np.dot(self.normal, np.cross(vec_pb, vec_pc)) / ABC_area
        beta = np.dot(self.normal, np.cross(vec_pc, vec_pa)) / ABC_area
        gamma = 1.0 - alpha - beta

        # if alpha + beta are bigger than 1 then gamma is negative, if alpha or beta are smaller than 0 they're outside the triangle
        if alpha < -EPSILON or beta < -EPSILON or gamma < -EPSILON:
            return None            
        
        return t, self
    
    def get_normal(self, hit_pt):
        return self.normal

        

class Diamond(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        l = [Triangle(self.v_list[i], self.v_list[j], self.v_list[k]) for i,j,k in t_idx]
        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(
                self.ambient,
                self.diffuse,
                self.specular,
                self.shininess,
                self.reflection,
            )

    def intersect(self, ray: Ray):
        nearest_obj = None
        t = np.inf

        for triangle in self.triangle_list:
            hit = triangle.intersect(ray)
            if hit is not None:
                dist, obj = hit
                if dist < t:
                    t = dist
                    nearest_obj = obj

        if nearest_obj is None:
            return None
        else:
            return t, nearest_obj

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        # As noted in the lecture, we get a quadratic equation of the form at^2 + 2bt + c
        a = np.dot(ray.direction, ray.direction) #quadratic coefficient

        origin_center_vec = ray.origin - self.center
        b = 2.0 * np.dot(ray.direction, origin_center_vec) # linear coefficient

        c = np.dot(origin_center_vec, origin_center_vec) - (self.radius ** 2) # constant

        # Using standard quadratic equation roots formula
        discriminant = (b ** 2.0) - (4.0 * a * c)
        if discriminant < 0: # No solutions
            return None
        
        discriminant_root = np.sqrt(discriminant)

        t1 = (-b - discriminant_root) / (2.0 * a)
        t2 = (-b + discriminant_root) / (2.0 * a)

        # t1 is smaller and therefore always closer than t2 if non-negative
        if t1 > EPSILON:
            return t1, self
        
        # this will only happen if t1 is sitting behind the camera (negative t1)
        if t2 > EPSILON:
            return t2, self
        
        return None
    
    def get_normal (self, hit_pt):
        return normalize(hit_pt - self.center)
        

