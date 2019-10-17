# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import vks.vulkanbuffer
import vks.vulkanexamplebase
import vks.vulkanglobals
import vks.vulkantexture

import glm
import numpy as np

VERTEX_BUFFER_BIND_ID = 0

class VulkanExample(vks.vulkanexamplebase.VulkanExampleBase):
    def __init__(self):
        super().__init__(enableValidation=True)
        self.texture = vks.vulkantexture.Texture2D()
        self.vertices = {'inputState': None, 'bindingDescriptions': [], 'attributeDescriptions': []}
        self.vertexBuffer = None
        self.vertexShape = None
        self.indexBuffer = None
        self.indexCount = 0
        self.uniformBufferVS = None
        self.uboVS = {'projection': glm.mat4(), 'model': glm.mat4(), 'viewPos': glm.vec4(), 'loadBias': glm.vec1(0.0)}
        self.pipelines = {'solid': None}
        self.pipelineLayout = None
        self.descriptorSetLayout = None
        self.descriptorSet = None
        self.zoom = -2.5
        self.title = "Vulkan Example - Texture loading"
        self.rotation = glm.vec3(0.0, 15.0, 0.0)
        self.settings['overlay'] = True

    def __del__(self):
        self.texture.destroy()
        if self.pipelines['solid']:
            vk.vkDestroyPipeline(self.device, self.pipelines['solid'], None)
        if self.pipelineLayout:
            vk.vkDestroyPipelineLayout(self.device, self.pipelineLayout, None)
        if self.descriptorSetLayout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, None)
        if self.vertexBuffer:
            self.vertexBuffer.destroy()
        if self.indexBuffer:
            self.indexBuffer.destroy()
        if self.uniformBufferVS:
            self.uniformBufferVS.destroy()

    def getEnabledFeatures(self):
        # Enable anisotropic filtering if supported
        if (self.deviceFeatures.samplerAnisotropy):
            self.enabledFeatures.samplerAnisotropy = vk.VK_TRUE

    def loadTexture(self):
        filename = self.getAssetPath() + "textures/metalplate01_rgba.ktx"
        self.texture.loadFromFile(filename, vk.VK_FORMAT_R8G8B8A8_UNORM, self.vulkanDevice, self.queue)

    def generateQuad(self):
        self.vertexShape = np.dtype([('pos', np.float32, (3,)), ('uv', np.float32, (2,)), ('normal', np.float32, (3,))]) #position, uv, normal
        # Setup vertices for a single uv-mapped quad made from two triangles
        vertexdata = np.array(
            [
            ( [  1.0,  1.0, 0.0 ], [ 1.0, 1.0 ],[ 0.0, 0.0, 1.0 ] ),
            ( [ -1.0,  1.0, 0.0 ], [ 0.0, 1.0 ],[ 0.0, 0.0, 1.0 ] ),
            ( [ -1.0, -1.0, 0.0 ], [ 0.0, 0.0 ],[ 0.0, 0.0, 1.0 ] ),
            ( [  1.0, -1.0, 0.0 ], [ 1.0, 0.0 ],[ 0.0, 0.0, 1.0 ] )
            ],
        dtype=self.vertexShape)
        vertexBufferSize = vertexdata.size * vertexdata.itemsize

        indexdata = np.array([ 0, 1, 2 , 2, 3, 0 ], dtype = np.uint32)
        indexBufferSize = indexdata.size * indexdata.itemsize
        self.indexCount = len(indexdata)

        # Create buffers
        # For the sake of simplicity we won't stage the vertex data to the gpu memory
        # Vertex buffer
        self.vertexBuffer = self.vulkanDevice.createvksBuffer(vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize, vertexdata
        )
        # Index buffer
        self.indexBuffer = self.vulkanDevice.createvksBuffer(vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            indexBufferSize, indexdata
        )

    def setupVertexDescriptions(self):
        vertexInputBinding = vk.VkVertexInputBindingDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            stride = self.vertexShape.itemsize,
            inputRate = vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        self.vertices['bindingDescriptions'].append(vertexInputBinding)
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 0,
            # Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['pos'][1] #  offsetof(vertexShape, pos)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 1,
            # UV attribute is two 32 bit signed (SFLOAT) floats (R32 G32)
            format = vk.VK_FORMAT_R32G32_SFLOAT,
            offset = self.vertexShape.fields['uv'][1] #  offsetof(vertexShape, pos)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 2,
            # Normal attribute is threeo 32 bit signed (SFLOAT) floats (R32 G32 B32)
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['normal'][1] #  offsetof(vertexShape, pos)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        self.vertices['inputState'] = vk.VkPipelineVertexInputStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount = len(self.vertices['bindingDescriptions']),
            pVertexBindingDescriptions = self.vertices['bindingDescriptions'],
            vertexAttributeDescriptionCount = len(self.vertices['attributeDescriptions']),
            pVertexAttributeDescriptions = self.vertices['attributeDescriptions']
        )

    def prepareUniformBuffers(self):
        uboVSSize = sum([glm.sizeof(ubo) for ubo in self.uboVS.values()])
        self.uniformBufferVS = self.vulkanDevice.createvksBuffer(
            vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uboVSSize
        )
        self.updateUniformBuffers()

    def updateUniformBuffers(self):
        self.uboVS['projection'] = glm.perspective(glm.radians(60.0), self.width / self.height, 0.001, 256.0)
        view = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, self.zoom))
        self.uboVS['model'] = view * glm.translate(glm.mat4(1.0), self.cameraPos)
        self.uboVS['model'] = glm.rotate(self.uboVS['model'], glm.radians(self.rotation.x), glm.vec3(1.0, 0.0, 0.0))
        self.uboVS['model'] = glm.rotate(self.uboVS['model'], glm.radians(self.rotation.y), glm.vec3(0.0, 1.0, 0.0))
        self.uboVS['model'] = glm.rotate(self.uboVS['model'], glm.radians(self.rotation.z), glm.vec3(0.0, 0.0, 1.0))
        self.uboVS['viewPos'] = glm.vec4(0.0, 0.0, -self.zoom, 0.0)

        uboVSSize = sum([glm.sizeof(ubo) for ubo in self.uboVS.values()])
        uboVSBuffer = np.concatenate((
            np.array(self.uboVS['projection']).flatten(order='C'),
            np.array(self.uboVS['model']).flatten(order='C'),
            np.array(self.uboVS['viewPos']).flatten(order='C'),
            np.array(self.uboVS['loadBias']).flatten(order='C')
        ))
        self.uniformBufferVS.map()
        self.uniformBufferVS.copyTo(uboVSBuffer, uboVSSize)
        self.uniformBufferVS.unmap()

    def setupDescriptorSetLayout(self):
        setLayoutBindings = []
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_VERTEX_BIT,
            binding = 0
        )
        setLayoutBindings.append(layoutBinding)
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            binding = 1
        )
        setLayoutBindings.append(layoutBinding)
        descriptorLayout = vk.VkDescriptorSetLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = None,
            bindingCount = len(setLayoutBindings),
            pBindings = setLayoutBindings
        )
        self.descriptorSetLayout = vk.vkCreateDescriptorSetLayout(self.device, descriptorLayout, None)
        pPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext = None,
            setLayoutCount = 1,
            pSetLayouts = [self.descriptorSetLayout]
        )
        self.pipelineLayout = vk.vkCreatePipelineLayout(self.device, pPipelineLayoutCreateInfo, None)

    def preparePipelines(self):
        inputAssemblyState = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology = vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            flags= 0,
            primitiveRestartEnable = vk.VK_FALSE
        )
        rasterizationState = vk.VkPipelineRasterizationStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = vk.VK_POLYGON_MODE_FILL,
            cullMode = vk.VK_CULL_MODE_NONE,
            frontFace = vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            flags = 0,
            depthClampEnable = vk.VK_FALSE,
            lineWidth = 1.0
        )
        
    def prepare(self):
        super().prepare()
        self.loadTexture()
        self.generateQuad()
        self.setupVertexDescriptions()
        self.prepareUniformBuffers()
        self.setupDescriptorSetLayout()
        self.preparePipelines()

texture = VulkanExample()
texture.main()
