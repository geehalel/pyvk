# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import vks.vulkanbuffer
import vks.vulkancamera
import vks.vulkanexamplebase
import vks.vulkanglobals
import vks.vulkantexture

import glm
import numpy as np
import imgui
import pyassimp
# no material.py in Fedora! copy by hand
import pyassimp.material

VERTEX_BUFFER_BIND_ID = 0

# From /usr/include/assimp/material.h
# pyassimp does not define them
AI_MATKEY_NAME = ('name', 0)
AI_MATKEY_COLOR_AMBIENT = ('ambient', 0)
AI_MATKEY_COLOR_DIFFUSE = ('diffuse', 0)
AI_MATKEY_COLOR_SPECULAR = ('specular', 0)
AI_MATKEY_OPACITY = ('opacity', 0)

class Scene:

    def __init__(self, vkdevice, queue):
        self.vulkanDevice = vkdevice
        self.queue = queue
        self.descriptorPool = None
        # We will be using separate descriptor sets (and bindings)
        # for material and scene related uniforms
        self.descriptorSetLayouts = {'material': None, 'scene': None}
        # We will be using one single index and vertex buffer
        # containing vertices and indices for all meshes in the scene
        # This allows us to keep memory allocations down
        self.vertexBuffer = None
        self.indexBuffer = None

        self.descriptorSetScene = None
        self.aScene = None

        self.assetPath=""
        # Shader properites for a material
        # Will be passed to the shaders using push constant
        self.scenematerialShape = {'ambient': glm.vec4(), 'diffuse': glm.vec4(),'specular': glm.vec4(), 'opacity': glm.vec1(0.0)}
        self.materials = []
        self.meshes = []
        # Shared ubo containing matrices used by all
        # materials and meshes
        self.uniformBuffer = None
        self.uniformData = {'projection': glm.mat4(), 'view': glm.mat4(), 'model': glm.mat4(), 'lightPos': glm.vec4(1.25, 8.35, 0.0, 0.0)}
        # Scene uses multiple pipelines
        self.pipelines = {'solid': None, 'blending': None, 'wireframe': None}
        # Shared pipeline layout
        self.pipelineLayout = None
        # For displaying only a single part of the scene
        self.renderSingleScenePart = False
        self.scenePartIndex = 0

        # TODO if that does not work, do not use createvksBuffer but do it by hand as in the demo
        uboSize = sum([glm.sizeof(ubo) for ubo in self.uniformData.values()])
        self.uniformBuffer = self.vulkanDevice.createvksBuffer(
            vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uboSize
        )
        self.uniformBuffer.map()

    def __del__(self):
        self.vertexBuffer.destroy()
        self.indexBuffer.destroy()
        for material in self.materials:
            material['diffuse'].destroy()
        vk.vkDestroyPipelineLayout(self.vulkanDevice.logicalDevice, self.pipelineLayout, None)
        vk.vkDestroyDescriptorSetLayout(self.vulkanDevice.logicalDevice, self.descriptorSetLayouts['material'], None)
        vk.vkDestroyDescriptorSetLayout(self.vulkanDevice.logicalDevice, self.descriptorSetLayouts['scene'], None)
        vk.vkDestroyDescriptorPool(self.vulkanDevice.logicalDevice, self.descriptorPool, None)
        vk.vkDestroyPipeline(self.vulkanDevice.logicalDevice, self.pipelines['solid'], None)
        vk.vkDestroyPipeline(self.vulkanDevice.logicalDevice, self.pipelines['blending'], None)
        vk.vkDestroyPipeline(self.vulkanDevice.logicalDevice, self.pipelines['wireframe'], None)
        self.uniformBuffer.destroy()

    # Get materials from the assimp scene and map to our scene structures
    def loadMaterials(self):
        for i in range(len(self.aScene.materials)):
            m = self.aScene.materials[i]
            scenematerial = {
                'name': None,
                'properties': {'ambient': glm.vec4(), 'diffuse': glm.vec4(),'specular': glm.vec4(), 'opacity': 0.0},
                'diffuse': vks.vulkantexture.Texture2D(),
                'descriptorSet': None,
                'pipeline': None
                }
            scenematerial['name'] = m.properties.get(AI_MATKEY_NAME)
            color = m.properties.get(AI_MATKEY_COLOR_AMBIENT) # returned as a list r, g, b, a hopefully
            scenematerial['properties']['ambient'] = glm.vec4(color[0]) + glm.vec4(0.1)
            color = m.properties.get(AI_MATKEY_COLOR_DIFFUSE)
            scenematerial['properties']['diffuse'] = glm.vec4(color[0])
            color = m.properties.get(AI_MATKEY_COLOR_SPECULAR)
            scenematerial['properties']['specular'] = glm.vec4(color[0])
            if  m.properties.get(AI_MATKEY_OPACITY):
                scenematerial['properties']['opacity'] = m.properties.get(AI_MATKEY_OPACITY)
            if scenematerial['properties']['opacity'] > 0.0:
                scenematerial['properties']['specular'] = glm.vec4(0.0)
            print("Material: \"" + scenematerial['name'] + "\"")

            # Textures
            texFormatSuffix = ""
            texFormat = None
            # Get supported compressed texture format
            if self.vulkanDevice.features.textureCompressionBC:
                texFormatSuffix = "_bc3_unorm"
                texFormat = vk.VK_FORMAT_BC3_UNORM_BLOCK
            elif self.vulkanDevice.features.textureCompressionASTC_LDR:
                texFormatSuffix = "_astc_8x8_unorm"
                texFormat = vk.VK_FORMAT_ASTC_8x8_UNORM_BLOCK
            elif self.vulkanDevice.features.textureCompressionETC2:
                texFormatSuffix = "_etc2_unorm"
                texFormat = vk.VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
            else:
                raise NotImplementedError("Device does not support any compressed texture format!")

            if m.properties.get(('file', pyassimp.material.aiTextureType_DIFFUSE)):
                texturefile = m.properties.get(('file', pyassimp.material.aiTextureType_DIFFUSE))
                print("  Diffuse: \"" + texturefile + "\"")
                texturefile = texturefile.replace("\\", "/")
                suffixidx = texturefile.find(".ktx")
                filename = texturefile[:suffixidx] + texFormatSuffix + texturefile[suffixidx:]
                scenematerial['diffuse'].loadFromFile(self.assetPath + filename, texFormat, self.vulkanDevice, self.queue)
            else:
                print("  Material has no diffuse, using dummy texture!")
                scenematerial['diffuse'].loadFromFile(self.assetPath + "dummy_rgba_unorm.ktx", vk.VK_FORMAT_R8G8B8A8_UNORM, self.vulkanDevice, self.queue)
            # For scenes with multiple textures per material we would need to check for additional texture types, e.g.:
            # aiTextureType_HEIGHT, aiTextureType_OPACITY, aiTextureType_SPECULAR, etc.

            # Assign pipeline
            scenematerial['pipeline'] = self.pipelines['solid'] if scenematerial['properties']['opacity'] == 0.0 else self.pipelines['solid']

            self.materials.append(scenematerial)

        # Generate descriptor sets for the materials
        # Descriptor pool
        poolSizes = []
        poolSize = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = len(self.materials)
        )
        poolSizes.append(poolSize)
        poolSize = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = len(self.materials)
        )
        poolSizes.append(poolSize)
        descriptorPoolInfo = vk.VkDescriptorPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = len(poolSizes),
            pPoolSizes = poolSizes,
            maxSets = len(self.materials) + 1
        )
        self.descriptorPool = vk.vkCreateDescriptorPool(self.vulkanDevice.logicalDevice, descriptorPoolInfo, None)
        # Descriptor set and pipeline layouts
        # Set 0: Scene matrices
        setLayoutBindings = []
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_VERTEX_BIT,
            binding = 0
        )
        setLayoutBindings.append(layoutBinding)
        descriptorLayout = vk.VkDescriptorSetLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = len(setLayoutBindings),
            pBindings = setLayoutBindings
        )
        self.descriptorSetLayouts['scene'] = vk.vkCreateDescriptorSetLayout(self.vulkanDevice.logicalDevice, descriptorLayout, None)
        # Set 1: Material data
        setLayoutBindings = []
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            binding = 0
        )
        setLayoutBindings.append(layoutBinding)
        descriptorLayout = vk.VkDescriptorSetLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = len(setLayoutBindings),
            pBindings = setLayoutBindings
        )
        self.descriptorSetLayouts['material'] = vk.vkCreateDescriptorSetLayout(self.vulkanDevice.logicalDevice, descriptorLayout, None)
        # Setup pipeline layout
        setLayouts = [self.descriptorSetLayouts['scene'], self.descriptorSetLayouts['material']]
        scenematerialSize = sum([glm.sizeof(scenematerial) for scenematerial in self.scenematerialShape.values()])
        pushConstantRange = vk.VkPushConstantRange(
            stageFlags = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            offset = 0,
            size = scenematerialSize
        )
        pPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = len(setLayouts),
            pSetLayouts = setLayouts,
            pushConstantRangeCount = 1,
            pPushConstantRanges = [ pushConstantRange ]
        )
        self.pipelineLayout = vk.vkCreatePipelineLayout(self.vulkanDevice.logicalDevice, pPipelineLayoutCreateInfo, None)
        # Material descriptor sets
        for m in self.materials:
            allocInfo = vk.VkDescriptorSetAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool = self.descriptorPool,
                descriptorSetCount = 1,
                pSetLayouts = [self.descriptorSetLayouts['material']]
            )
            descriptorSets = vk.vkAllocateDescriptorSets(self.vulkanDevice.logicalDevice, allocInfo)
            m['descriptorSet'] = descriptorSets[0]
            # Binding 0: Diffuse texture
            writeDescriptorSets = []
            writeDescriptorSet = vk.VkWriteDescriptorSet(
                sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet = m['descriptorSet'],
                descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                dstBinding = 0,
                pImageInfo = [ m['diffuse'].descriptor ],
                descriptorCount = 1
            )
            writeDescriptorSets.append(writeDescriptorSet)
            vk.vkUpdateDescriptorSets(self.vulkanDevice.logicalDevice, len(writeDescriptorSets), writeDescriptorSets, 0, None)
        # Scene descriptor set
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = self.descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = [self.descriptorSetLayouts['scene']]
        )
        descriptorSets = vk.vkAllocateDescriptorSets(self.vulkanDevice.logicalDevice, allocInfo)
        self.descriptorSetScene = descriptorSets[0] 
        writeDescriptorSets = []
        writeDescriptorSet = vk.VkWriteDescriptorSet(
            sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = self.descriptorSetScene,
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            dstBinding = 0,
            pBufferInfo = [ self.uniformBuffer.descriptor ],
            descriptorCount = 1
        )
        writeDescriptorSets.append(writeDescriptorSet)
        vk.vkUpdateDescriptorSets(self.vulkanDevice.logicalDevice, len(writeDescriptorSets), writeDescriptorSets, 0, None)

    # Load all meshes from the scene and generate the buffers for rendering them
    def loadMeshes(self, copyCmd):
        pass

    def load(self, filename, copyCmd):
        flags = pyassimp.postprocess.aiProcess_PreTransformVertices | pyassimp.postprocess.aiProcess_Triangulate | pyassimp.postprocess.aiProcess_GenNormals
        try:
            self.aScene = pyassimp.load(filename, processing = flags)
        except pyassimp.AssimpError as e:
            print('Error parsing', filename, ':', e)
            raise RuntimeError
        if self.aScene is not None:
            self.loadMaterials()
            self.loadMeshes(copyCmd)
        else:
            print('Error parsing', filename)
            raise RuntimeError

class VulkanExample(vks.vulkanexamplebase.VulkanExampleBase):
    """
Summary:
* Renders a scene made of multiple parts with different materials and textures.
*
* The example loads a scene made up of multiple parts into one vertex and index buffer to only
* have one (big) memory allocation. In Vulkan it's advised to keep number of memory allocations
* down and try to allocate large blocks of memory at once instead of having many small allocations.
*
* Every part has a separate material and multiple descriptor sets (set = x layout qualifier in GLSL)
* are used to bind a uniform buffer with global matrices and the part's material's sampler at once.
*
* To demonstrate another way of passing data the example also uses push constants for passing
* material properties.
*
* Note that this example is just one way of rendering a scene made up of multiple parts in Vulkan.
    """
    def __init__(self):
        super().__init__(enableValidation=True)
        self.vertexShape = np.dtype([('pos', np.float32, (3,)), ('normal', np.float32, (3,)), ('uv', np.float32, (2,)), ('color', np.float32, (3,))]) #position, normal, uv, color
        self.vertices = {'inputState': None, 'bindingDescriptions': [], 'attributeDescriptions': []}

        self.title = "Multi-part scene rendering"
        self.rotationSpeed = 0.5
        self.camera.type = vks.vulkancamera.CameraType.firstperson
        self.camera.movementSpeed = 7.5
        self.camera.position = glm.vec3(15.0, -13.5, 0.0)
        self.camera.setRotation(glm.vec3(5.0, 90.0, 0.0))
        self.camera.setPerspective(60.0, self.width / self.height, 0.1, 256.0)
        self.settings['overlay'] = True

    def setupVertexDescriptions(self):
        vertexInputBinding = vk.VkVertexInputBindingDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            stride = self.vertexShape.itemsize,
            inputRate = vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        self.vertices['bindingDescriptions'].append(vertexInputBinding)
        # Location 0 : Position
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 0,
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['pos'][1] #  offsetof(vertexShape, pos)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        # Location 1 : Normal
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 1,
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['normal'][1] #  offsetof(vertexShape, normal)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        # Location 2 : Texture coordinates
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 2,
            format = vk.VK_FORMAT_R32G32_SFLOAT,
            offset = self.vertexShape.fields['uv'][1] #  offsetof(vertexShape, uv)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        # Location 3 : Color
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 3,
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['color'][1] #  offsetof(vertexShape, color)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)

        self.vertices['inputState'] = vk.VkPipelineVertexInputStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount = len(self.vertices['bindingDescriptions']),
            pVertexBindingDescriptions = self.vertices['bindingDescriptions'],
            vertexAttributeDescriptionCount = len(self.vertices['attributeDescriptions']),
            pVertexAttributeDescriptions = self.vertices['attributeDescriptions']
        )

    def loadScene(self):
        copyCmd = super().createCommandBuffer(vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, False)
        scene = Scene(self.vulkanDevice, self.queue)
        scene.assetPath = self.getAssetPath() + "models/sibenik/"
        scene.load(self.getAssetPath() + "models/sibenik/sibenik.dae", copyCmd)
        vk.vkFreeCommandBuffers(self.device, self.cmdPool, 1, copyCmd)


    def prepare(self):
        super().prepare()
        self.setupVertexDescriptions()
        self.loadScene()

    def render(self):
        if not self.prepared:
            return
        #vk.vkDeviceWaitIdle(self.device)
        self.draw()

    def viewChanged(self):
        self.updateUniformBuffers()

scenerendering = VulkanExample()
scenerendering.main()
