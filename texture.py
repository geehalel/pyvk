# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import vks.vulkanbuffer
import vks.vulkanexamplebase
import vks.vulkanglobals
import vks.vulkantexture

import glm
import numpy as np
import imgui

VERTEX_BUFFER_BIND_ID = 0

class VulkanExample(vks.vulkanexamplebase.VulkanExampleBase):
    def __init__(self):
        super().__init__(enableValidation=False)
        self.texture = vks.vulkantexture.Texture2D()
        self.vertices = {'inputState': None, 'bindingDescriptions': [], 'attributeDescriptions': []}
        self.vertexBuffer = None
        self.vertexShape = None
        self.indexBuffer = None
        self.indexCount = 0
        self.uniformBufferVS = None
        self.uboVS = {'projection': glm.mat4(), 'model': glm.mat4(), 'viewPos': glm.vec4(), 'lodBias': glm.vec1(0.0)}
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
            np.array(self.uboVS['lodBias']).flatten(order='C')
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
        blendAttachmentState = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask = 0xf,
            blendEnable = vk.VK_FALSE
        )
        colorBlendState = vk.VkPipelineColorBlendStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount = 1,
            pAttachments = [blendAttachmentState]
        )
        opState = vk.VkStencilOpState(
            #failOp = vk.VK_STENCIL_OP_KEEP,
            #passOp = vk.VK_STENCIL_OP_KEEP,
            compareOp = vk.VK_COMPARE_OP_ALWAYS
        )
        depthStencilState = vk.VkPipelineDepthStencilStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable = vk.VK_TRUE,
            depthWriteEnable = vk.VK_TRUE,
            depthCompareOp = vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            #depthBoundsTestEnable = vk.VK_FALSE,
            #stencilTestEnable = vk.VK_FALSE,
            front = opState,
            back = opState
        )
        viewportState = vk.VkPipelineViewportStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount = 1,
            scissorCount = 1,
            flags = 0
        )
        multisampleState = vk.VkPipelineMultisampleStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples = vk.VK_SAMPLE_COUNT_1_BIT,
            flags = 0
        )
        dynamicStateEnables = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        dynamicState = vk.VkPipelineDynamicStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount = len(dynamicStateEnables),
            pDynamicStates = dynamicStateEnables,
            flags = 0
        )
        shaderStages = []
        # Vertex shader
        shaderStage = vk.VkPipelineShaderStageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage = vk.VK_SHADER_STAGE_VERTEX_BIT,
            module = vks.vulkantools.loadShader(self.getAssetPath() + "shaders/texture/texture.vert.spv", self.device),
            pName = "main"
        )
        shaderStages.append(shaderStage)
        # Fragment shader
        shaderStage = vk.VkPipelineShaderStageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            module = vks.vulkantools.loadShader(self.getAssetPath() + "shaders/texture/texture.frag.spv", self.device),
            pName = "main"
        )
        shaderStages.append(shaderStage)
        pipelineCreateInfo = vk.VkGraphicsPipelineCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            layout = self.pipelineLayout,
            renderPass = self.renderPass,
            pVertexInputState = self.vertices['inputState'],
            pInputAssemblyState = inputAssemblyState,
            pRasterizationState = rasterizationState,
            pColorBlendState = colorBlendState,
            pMultisampleState = multisampleState,
            pViewportState = viewportState,
            pDepthStencilState = depthStencilState,
            pDynamicState = dynamicState,
            stageCount = len(shaderStages),
            pStages = shaderStages,
            flags = 0,
            basePipelineIndex = -1,
            basePipelineHandle = vk.VK_NULL_HANDLE
        )
        pipe = vk.vkCreateGraphicsPipelines(self.device, self.pipelineCache, 1, [pipelineCreateInfo], None)
        try:
            self.pipelines['solid'] = pipe[0]
        except TypeError:
            self.pipelines['solid'] = pipe

    def setupDescriptorPool(self):
        poolSizes = []
        poolSize = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1
        )
        poolSizes.append(poolSize)
        poolSize = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = 1
        )
        poolSizes.append(poolSize)
        descriptorPoolInfo = vk.VkDescriptorPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = len(poolSizes),
            pPoolSizes = poolSizes,
            maxSets = 2
        )
        self.descriptorPool = vk.vkCreateDescriptorPool(self.device, descriptorPoolInfo, None)

    def setupDescriptorSet(self):
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = self.descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = [self.descriptorSetLayout]
        )
        descriptorSets = vk.vkAllocateDescriptorSets(self.device, allocInfo)
        self.descriptorSet = descriptorSets[0]
        # Setup a descriptor image info for the current texture to be used as a combined image sampler
        textureDescriptor = vk.VkDescriptorImageInfo(
            sampler = self.texture.sampler,     # The sampler (Telling the pipeline how to sample the texture, including repeat, border, etc.)
            imageView = self.texture.view,      # The image's view (images are never directly accessed by the shader, but rather through views defining subresources)
            imageLayout = self.texture.imageLayout # The current layout of the image (Note: Should always fit the actual use, e.g. shader read)
        )
        writeDescriptorSets = []
        # Binding 0 : Vertex shader uniform buffer
        writeDescriptorSet = vk.VkWriteDescriptorSet(
            sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = self.descriptorSet,
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            dstBinding = 0,
            pBufferInfo = [ self.uniformBufferVS.descriptor ],
            descriptorCount = 1
        )
        writeDescriptorSets.append(writeDescriptorSet)
        # Binding 1 : Fragment shader texture sampler
        # Fragment shader: layout (binding = 1) uniform sampler2D samplerColor
        writeDescriptorSet = vk.VkWriteDescriptorSet(
            sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = self.descriptorSet,
            descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, # // The descriptor set will use a combined image sampler (sampler and image could be split)
            dstBinding = 1,                       # Shader binding point 1
            pImageInfo = [ textureDescriptor ],   # Pointer to the descriptor image for our texture
            descriptorCount = 1
        )
        writeDescriptorSets.append(writeDescriptorSet)
        vk.vkUpdateDescriptorSets(self.device, len(writeDescriptorSets), writeDescriptorSets, 0, None)

    def buildCommandBuffers(self):
        cmdBufInfo = vk.VkCommandBufferBeginInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = None
        )
        clearValues = []
        clearValue = vk.VkClearValue(
            color = self.defaultClearColor
        )
        clearValues.append(clearValue)
        clearValue = vk.VkClearValue(
            depthStencil = [1.0, 0 ]
        )
        clearValues.append(clearValue)
        offset = vk.VkOffset2D(x = 0, y = 0)
        extent = vk.VkExtent2D(width = self.width, height = self.height)
        renderArea = vk.VkRect2D(offset = offset, extent = extent)
        renderPassBeginInfo = vk.VkRenderPassBeginInfo(
            sType = vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext = None,
            renderPass = self.renderPass,
            renderArea = renderArea,
            clearValueCount = 2,
            pClearValues = clearValues,
        )
        for i in range(len(self.drawCmdBuffers)):
            renderPassBeginInfo.framebuffer = self.frameBuffers[i]
            vk.vkBeginCommandBuffer(self.drawCmdBuffers[i], cmdBufInfo)
            vk.vkCmdBeginRenderPass(self.drawCmdBuffers[i], renderPassBeginInfo, vk.VK_SUBPASS_CONTENTS_INLINE)
            viewport = vk.VkViewport(
                height = float(self.height),
                width = float(self.width),
                minDepth = 0.0,
                maxDepth = 1.0
            )
            vk.vkCmdSetViewport(self.drawCmdBuffers[i], 0, 1, [viewport])
            # Update dynamic scissor state
            offsetscissor = vk.VkOffset2D(x = 0, y = 0)
            extentscissor = vk.VkExtent2D(width = self.width, height = self.height)
            scissor = vk.VkRect2D(offset = offsetscissor, extent = extentscissor)
            vk.vkCmdSetScissor(self.drawCmdBuffers[i], 0, 1, [scissor])

            vk.vkCmdBindDescriptorSets(self.drawCmdBuffers[i], vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout, 0, 1, [self.descriptorSet], 0, None)
            vk.vkCmdBindPipeline(self.drawCmdBuffers[i], vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelines['solid'])

            offsets = [ 0 ]
            vk.vkCmdBindVertexBuffers(self.drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, [self.vertexBuffer.buffer], offsets)
            vk.vkCmdBindIndexBuffer(self.drawCmdBuffers[i], self.indexBuffer.buffer, 0, vk.VK_INDEX_TYPE_UINT32)
            # Draw indexed triangle
            vk.vkCmdDrawIndexed(self.drawCmdBuffers[i], self.indexCount, 1, 0, 0, 0)
            self.drawUI(self.drawCmdBuffers[i])
            vk.vkCmdEndRenderPass(self.drawCmdBuffers[i])
            vk.vkEndCommandBuffer(self.drawCmdBuffers[i])

    def prepare(self):
        super().prepare()
        self.loadTexture()
        self.generateQuad()
        self.setupVertexDescriptions()
        self.prepareUniformBuffers()
        self.setupDescriptorSetLayout()
        self.preparePipelines()
        self.setupDescriptorPool()
        self.setupDescriptorSet()
        self.buildCommandBuffers()
        # self.submitInfo = vk.VkSubmitInfo(sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        #     pWaitDstStageMask = self.submitPipelineStages,
        #     waitSemaphoreCount = 1,
        #     pWaitSemaphores = [self.semaphores['presentComplete']],
        #     signalSemaphoreCount = 1,
        #     pSignalSemaphores = [self.semaphores['renderComplete']],
        #     # To be able to set pCommandBuffers directly in draw()
        #     commandBufferCount = 1,
        #     pCommandBuffers = [ self.drawCmdBuffers[0] ]
        # )
        # # optimization to avoid creating a new array each time
        # self.submit_list = vk.ffi.new('VkSubmitInfo[1]', [self.submitInfo])
        self.prepared = True

    def draw(self):
        super().prepareFrame()
        # self.submitInfo.commandBufferCount = 1
        # TODO try to avoid creating submitInfo at each frame
        # need to get CData pointer on drawCmdBuffers[*]
        # self.submitInfo.pCommandBuffers[0] = self.drawCmdBuffers[self.currentBuffer]
        # vk.vkQueueSubmit(self.queue, 1, self.submit_list, vk.VK_NULL_HANDLE)
        submitInfo = vk.VkSubmitInfo(
            sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pWaitDstStageMask = [ vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT ],
            pWaitSemaphores = [ self.semaphores['presentComplete'] ],
            waitSemaphoreCount = 1,
            signalSemaphoreCount = 1,
            pSignalSemaphores = [ self.semaphores['renderComplete'] ],
            pCommandBuffers = [ self.drawCmdBuffers[self.currentBuffer] ],
            commandBufferCount = 1
        )
        vk.vkQueueSubmit(self.queue, 1,  submitInfo , vk.VK_NULL_HANDLE)
        super().submitFrame()

    def render(self):
        if not self.prepared:
            return
        # vk.vkDeviceWaitIdle(self.device)
        self.draw()

    def viewChanged(self):
        self.updateUniformBuffers()

    def onUpdateUIOverlay(self, overlay):
        #return
        if imgui.collapsing_header("Settings"):
            res, value = imgui.slider_float("LOD bias", self.uboVS['lodBias'].x, 0.0, self.texture.mipLevels)
            #res = False
            if res:
                overlay.updated = True
                self.uboVS['lodBias'].x = value
                self.updateUniformBuffers()


texture = VulkanExample()
texture.main()
