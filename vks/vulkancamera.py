# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import enum
import math

import glm
#define GLM_FORCE_RADIANS this is default now in glm and suppressed
#define GLM_FORCE_DEPTH_ZERO_TO_ONE use ortho/perspective*_ZO  (with ZO suffix)
# clip is between 0..+1.0 in Vulkan (-1.0..+1.0 in OpenGL)

class CameraType(enum.Enum):
    lookat = 0
    firstperson = 1

class Camera:
    def __init__(self):
        self.fov = 0.0
        self.znear = 0.0
        self.zfar = 0.0
        self.typecam = CameraType.lookat
        self.rotation = glm.vec3()
        self.position = glm.vec3()
        self.rotationSpeed = 1.0
        self.movementSpeed = 1.0
        self.updated = False
        self.matrices = {'perspective': None, 'view':None}
        self.keys = {'left': False, 'right': False, 'up': False, 'down': False}

    def updateViewMatrix(self):
        rotM = glm.mat4(1.0)

        rotM = glm.rotate(rotM, glm.radians(self.rotation.x), glm.vec3(1.0, 0.0, 0.0))
        rotM = glm.rotate(rotM, glm.radians(self.rotation.y), glm.vec3(0.0, 1.0, 0.0))
        rotM = glm.rotate(rotM, glm.radians(self.rotation.z), glm.vec3(0.0, 0.0, 1.0))

        transM = glm.translate(glm.mat4(1.0), self.position)

        if self.typecam == CameraType.firstperson:
            self.matrices['view'] = rotM * transM
        else:
            self.matrices['view'] = transM * rotM
        self.updated = True

    def moving(self):
        return True in self.keys.values()

    def update(self, deltaTime):
        self.updated = False
        if self.typecam == CameraType.firstperson:
            if self.moving():
                camFront = glm.vec3()
                camFront.x = -math.cos(glm.radians(self.rotation.x)) * math.sin(glm.radians(self.rotation.y))
                camFront.y =  math.sin(glm.radians(self.rotation.x))
                camFront.z = math.cos(glm.radians(self.rotation.x)) * math.cos(glm.radians(self.rotation.y))
                camFront = glm.normalize(camFront)

                moveSpeed = deltaTime * self.movementSpeed
                if self.keys['up']:
                    self.position += camFront * moveSpeed
                elif self.keys['down']:
                    self.position -= camFront * moveSpeed
                elif self.keys['left']:
                    self.position -= glm.normalize(glm.cross(camFront, glm.vec3(0.0, 1.0, 0.0))) * moveSpeed
                elif self.keys['right']:
                    self.position += glm.normalize(glm.cross(camFront, glm.vec3(0.0, 1.0, 0.0))) * moveSpeed

                self.updateViewMatrix()
    def rotate(self, delta):
        self.rotation += delta
        self.updateViewMatrix()
    def translate(self, delta):
        self.position += delta
        self.updateViewMatrix()
    def setRotation(self, rotation):
        self.rotation = rotation
        self.updateViewMatrix()
    def setPosition(self, position):
        self.position = position
        self.updateViewMatrix()
    def setPerspective(self, fov, aspect, znear, zfar):
        self.fov = fov
        self.znear = znear
        self.zfar = zfar
        self.matrices['perspective'] = glm.perspective(glm.radians(fov), aspect, znear, zfar)
