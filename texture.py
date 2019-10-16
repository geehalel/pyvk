# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import vks.vulkanexamplebase
import vks.vulkanglobals
import vks.vulkantexture

import glm
import numpy as np

class VulkanExample(vks.vulkanexamplebase.VulkanExampleBase):
    def __init__(self):
        super().__init__(enableValidation=True)
        self.texture = vks.vulkantexture.Texture2D()

    def loadTexture(self):
        filename = self.getAssetPath() + "textures/metalplate01_rgba.ktx"
        self.texture.loadFromFile(filename, vk.VK_FORMAT_R8G8B8A8_UNORM, self.vulkanDevice, self.queue)

    def prepare(self):
        super().prepare()
        self.loadTexture()

texture = VulkanExample()
texture.main()
