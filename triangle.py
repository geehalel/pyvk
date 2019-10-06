# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import vks.vulkanexamplebase
import vks.vulkanglobals
import glm
import numpy as np

import array

class VulkanExample(vks.vulkanexamplebase.VulkanExampleBase):
    def __init__(self):
        super().__init__(enableValidation=True)
        self.vertices = {'memory': None, 'buffer': None}
        self.indices = {'memory': None, 'buffer': None, 'count': 0}
        self.uniformBufferVS = {'memory': None, 'buffer': None, 'descriptor': {}}
        self.uboVS = {'projectionMatrix': glm.mat4(), 'modelMatrix': glm.mat4(), 'viewMatrix': glm.mat4()}
        self.pipelineLayout = None
        self.pipeline = None
        self.descriptorSetLayout = None
        self.descriptorSet = None
        self.zoom = -2.5
        self.title = "Vulkan Example - Basic indexed triangle"
        # uncomment for imgui support
        self.settings['overlay'] = True

    def __del__(self):
        vk.vkDestroyPipeline(self.device, self.pipeline, None)
        vk.vkDestroyPipelineLayout(self.device, self.pipelineLayout, None)
        vk.vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, None)

        vk.vkDestroyBuffer(self.device, self.vertices['buffer'], None)
        vk.vkFreeMemory(self.device, self.vertices['memory'], None)

        vk.vkDestroyBuffer(self.device, self.indices['buffer'], None)
        vk.vkFreeMemory(self.device, self.indices['memory'], None)

        vk.vkDestroyBuffer(self.device, self.uniformBufferVS['buffer'], None)
        vk.vkFreeMemory(self.device, self.uniformBufferVS['memory'], None)

		#vkDestroySemaphore(device, presentCompleteSemaphore, nullptr);
		#vkDestroySemaphore(device, renderCompleteSemaphore, nullptr);
		#for (auto& fence : waitFences)
		#	vkDestroyFence(device, fence, nullptr);

    def getCommandBuffer(self, begin):
        """
Get a new command buffer from the command pool
If begin is true, the command buffer is also started so we can start adding commands
        """
        cmdBufAllocateInfo = vk.VkCommandBufferAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool = self.cmdPool,
            level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount = 1
        )
        cmdBuffer = vk.vkAllocateCommandBuffers(self.device, cmdBufAllocateInfo)[0]

        if begin:
            cmdBufInfo = vk.VkCommandBufferBeginInfo(sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
            vk.vkBeginCommandBuffer(cmdBuffer, cmdBufInfo)
        return cmdBuffer

    def flushCommandBuffer(self, commandBuffer):
        """
End the command buffer and submit it to the queue
Uses a fence to ensure command buffer has finished executing before deleting it
        """
        assert(commandBuffer != vk.VK_NULL_HANDLE)
        vk.vkEndCommandBuffer(commandBuffer)

        submitInfo = vk.VkSubmitInfo(
            sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = [commandBuffer]
        )
        # Create fence to ensure that the command buffer has finished executing
        fenceCreateInfo = vk.VkFenceCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags = 0
        )
        fence = vk.vkCreateFence(self.device, fenceCreateInfo, None)
        # Submit to the queue
        vk.vkQueueSubmit(self.queue, 1, [submitInfo], fence)
        # Wait for the fence to signal that command buffer has finished executing
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, vks.vulkanglobals.DEFAULT_FENCE_TIMEOUT)

        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.cmdPool, 1, [commandBuffer])

    def prepareVertices(self, useStagingBuffers = True):
        self.vertexShape = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]], dtype = np.float32) # position,color
        vertexBuffer = np.array(
        [
            [ [  1.0,  1.0, 0.0 ], [ 1.0, 0.0, 0.0 ] ],
            [ [ -1.0,  1.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ],
            [ [  0.0, -1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ]
        ],
        dtype = np.float32)
        vertexBufferSize = vertexBuffer.size * vertexBuffer.itemsize

        indexBuffer = np.array([ 0, 1, 2 ], dtype = np.uint32)
        indexBufferSize = indexBuffer.size * indexBuffer.itemsize
        self.indices['count'] = len(indexBuffer)

        if (useStagingBuffers):
            stagingBuffers= { 'vertices': {'memory': None, 'buffer': None}, 'indices': {'memory': None, 'buffer': None}}
            # Vertex buffer
            #  Create a host-visible buffer to copy the vertex data to (staging buffer)
            vertexBufferInfo = vk.VkBufferCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = vertexBufferSize,
                usage = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            )
            stagingBuffers['vertices']['buffer'] = vk.vkCreateBuffer(self.device, vertexBufferInfo, None)
            # Request a host visible memory type that can be used to copy our data to
            # Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
            memReqs = vk.vkGetBufferMemoryRequirements(self.device, stagingBuffers['vertices']['buffer'])
            memAlloc = vk.VkMemoryAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            )
            stagingBuffers['vertices']['memory'] = vk.vkAllocateMemory(self.device, memAlloc, None)
            data = vk.vkMapMemory(self.device, stagingBuffers['vertices']['memory'], 0, memAlloc.allocationSize, 0)
            #memcpy
            # print(type(data), len(data)) # <class '_cffi_backend.buffer'>
            # should resize with nvidia drivers as data is greater than required
            #datawrapper = np.resize(np.array(data, copy=False), vertexBufferSize)
            #np.copyto(datawrapper,vertexBuffer.flatten(order='C').view(dtype=np.uint8), casting='no')
            datawrapper = np.array(data, copy=False)
            vbuf = np.resize(vertexBuffer.flatten(order='C').view(dtype=np.uint8), memReqs.size)
            np.copyto(datawrapper, vbuf, casting='no')
            vk.vkUnmapMemory(self.device, stagingBuffers['vertices']['memory'])
            vk.vkBindBufferMemory(self.device, stagingBuffers['vertices']['buffer'], stagingBuffers['vertices']['memory'], 0)

            #  Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
            vertexBufferInfo = vk.VkBufferCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = vertexBufferSize,
                usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            )
            self.vertices['buffer'] = vk.vkCreateBuffer(self.device, vertexBufferInfo, None)
            memReqs = vk.vkGetBufferMemoryRequirements(self.device, self.vertices['buffer'])
            memAlloc = vk.VkMemoryAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            )
            self.vertices['memory'] = vk.vkAllocateMemory(self.device, memAlloc, None)
            vk.vkBindBufferMemory(self.device, self.vertices['buffer'], self.vertices['memory'], 0)

            # Index buffer
            #  Create a host-visible buffer to copy the vertex data to (staging buffer)
            indexBufferInfo = vk.VkBufferCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = indexBufferSize,
                usage = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            )
            stagingBuffers['indices']['buffer'] = vk.vkCreateBuffer(self.device, indexBufferInfo, None)
            # Request a host visible memory type that can be used to copy our data to
            # Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
            memReqs = vk.vkGetBufferMemoryRequirements(self.device, stagingBuffers['indices']['buffer'])
            memAlloc = vk.VkMemoryAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            )
            stagingBuffers['indices']['memory'] = vk.vkAllocateMemory(self.device, memAlloc, None)
            data = vk.vkMapMemory(self.device, stagingBuffers['indices']['memory'], 0, memAlloc.allocationSize, 0)
            #memcpy
            #print(type(data)) # <class '_cffi_backend.buffer'>
            #datawrapper = np.resize(np.array(data, copy=False), indexBufferSize)
            #np.copyto(datawrapper, indexBuffer.flatten(order='C').view(dtype=np.uint8), casting='no')
            datawrapper = np.array(data, copy=False)
            vbuf = np.resize(indexBuffer.flatten(order='C').view(dtype=np.uint8), memReqs.size)
            np.copyto(datawrapper, vbuf, casting='no')
            vk.vkUnmapMemory(self.device, stagingBuffers['indices']['memory'])
            vk.vkBindBufferMemory(self.device, stagingBuffers['indices']['buffer'], stagingBuffers['indices']['memory'], 0)

            #  Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
            indexBufferInfo = vk.VkBufferCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = indexBufferSize,
                usage = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            )
            self.indices['buffer'] = vk.vkCreateBuffer(self.device, indexBufferInfo, None)
            memReqs = vk.vkGetBufferMemoryRequirements(self.device, self.indices['buffer'])
            memAlloc = vk.VkMemoryAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            )
            self.indices['memory'] = vk.vkAllocateMemory(self.device, memAlloc, None)
            vk.vkBindBufferMemory(self.device, self.indices['buffer'], self.indices['memory'], 0)

            # Buffer copies have to be submitted to a queue, so we need a command buffer for them
            # Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
            copyCmd = self.getCommandBuffer(True)
            # Put buffer region copies into command buffer
            copyRegion = vk.VkBufferCopy(size = vertexBufferSize)
            vk.vkCmdCopyBuffer(copyCmd, stagingBuffers['vertices']['buffer'], self.vertices['buffer'], 1, copyRegion)
            copyRegion = vk.VkBufferCopy(size = indexBufferSize)
            vk.vkCmdCopyBuffer(copyCmd, stagingBuffers['indices']['buffer'], self.indices['buffer'], 1, copyRegion)
            # Flushing the command buffer will also submit it to the queue and uses a fence to ensure that all commands have been executed before returning
            self.flushCommandBuffer(copyCmd)

            # Destroy staging buffers
            # Note: Staging buffer must not be deleted before the copies have been submitted and executed
            vk.vkDestroyBuffer(self.device, stagingBuffers['vertices']['buffer'], None)
            vk.vkFreeMemory(self.device, stagingBuffers['vertices']['memory'], None)
            vk.vkDestroyBuffer(self.device, stagingBuffers['indices']['buffer'], None)
            vk.vkFreeMemory(self.device, stagingBuffers['indices']['memory'], None)
        else:
            raise NotImplementedError('Host only visible buffers not implemented')

    def prepareUniformBuffers(self):
        """
Prepare and initialize a uniform buffer block containing shader uniforms
Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks
        """
        # Vertex shader uniform buffer block
        uboVSSize = sum([glm.sizeof(ubo) for ubo in self.uboVS.values()])
        bufferInfo = vk.VkBufferCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size = uboVSSize,
            # This buffer will be used as a uniform buffer
            usage = vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        )
        # Create a new buffer
        self.uniformBufferVS['buffer'] = vk.vkCreateBuffer(self.device, bufferInfo, None)
        # Get memory requirements including size, alignment and memory type
        memReqs = vk.vkGetBufferMemoryRequirements(self.device, self.uniformBufferVS['buffer'])
        # Get the memory type index that supports host visibile memory access
        # Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
        # We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
        #Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular base
        allocInfo = vk.VkMemoryAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext = None,
            allocationSize = memReqs.size,
            memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        )
        # Allocate memory for the uniform buffer
        self.uniformBufferVS['memory'] = vk.vkAllocateMemory(self.device, allocInfo, None)
        # Bind memory to buffer
        vk.vkBindBufferMemory(self.device, self.uniformBufferVS['buffer'], self.uniformBufferVS['memory'], 0)
        # Store information in the uniform's descriptor that is used by the descriptor set
        self.uniformBufferVS['descriptor'] = vk.VkDescriptorBufferInfo(
            buffer = self.uniformBufferVS['buffer'],
            offset = 0,
            range = uboVSSize
        )

        self.updateUniformBuffers()

    def updateUniformBuffers(self):
        self.uboVS['projectionMatrix'] = glm.perspective(glm.radians(60.0), self.width / self.height, 0.1, 256.0)
        self.uboVS['viewMatrix'] = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, self.zoom))
        self.uboVS['modelMatrix'] = glm.mat4(1.0)
        self.uboVS['modelMatrix'] = glm.rotate(self.uboVS['modelMatrix'], glm.radians(self.rotation.x), glm.vec3(1.0, 0.0, 0.0))
        self.uboVS['modelMatrix'] = glm.rotate(self.uboVS['modelMatrix'], glm.radians(self.rotation.y), glm.vec3(0.0, 1.0, 0.0))
        self.uboVS['modelMatrix'] = glm.rotate(self.uboVS['modelMatrix'], glm.radians(self.rotation.z), glm.vec3(0.0, 0.0, 1.0))

        uboVSSize = sum([glm.sizeof(ubo) for ubo in self.uboVS.values()])
        uboVSBuffer = np.concatenate((
            np.array(self.uboVS['projectionMatrix']).flatten(order='C'),
            np.array(self.uboVS['modelMatrix']).flatten(order='C'),
            np.array(self.uboVS['viewMatrix']).flatten(order='C')
        ))
        data = vk.vkMapMemory(self.device, self.uniformBufferVS['memory'], 0, uboVSSize, 0)
        datawrapper = np.array(data, copy=False)
        np.copyto(datawrapper, uboVSBuffer.view(dtype=np.uint8), casting='no')
        vk.vkUnmapMemory(self.device, self.uniformBufferVS['memory'])

    def setupDescriptorSetLayout(self):
        """
Setup layout of descriptors used in this example
Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
So every shader binding should map to one descriptor set layout binding
        """
        # Binding 0: Uniform buffer (Vertex shader)
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_VERTEX_BIT,
            pImmutableSamplers = None
        )
        descriptorLayout = vk.VkDescriptorSetLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = None,
            bindingCount = 1,
            pBindings = [layoutBinding]
        )
        self.descriptorSetLayout = vk.vkCreateDescriptorSetLayout(self.device, descriptorLayout, None)

        # Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
        # In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
        pPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext = None,
            setLayoutCount = 1,
            pSetLayouts = [self.descriptorSetLayout]
        )
        self.pipelineLayout = vk.vkCreatePipelineLayout(self.device, pPipelineLayoutCreateInfo, None)

    def preparePipelines(self):
        """
Create the graphics pipeline used in this example
Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
A pipeline is then stored and hashed on the GPU making pipeline changes very fast
Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)
        """

        # Construct the differnent states making up the pipeline

        # Input assembly state describes how primitives are assembled
        # This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
        inputAssemblyState = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology = vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )
        # Rasterization state
        rasterizationState = vk.VkPipelineRasterizationStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = vk.VK_POLYGON_MODE_FILL,
            cullMode = vk.VK_CULL_MODE_NONE,
            frontFace = vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthClampEnable = vk.VK_FALSE,
            rasterizerDiscardEnable = vk.VK_FALSE,
            depthBiasEnable = vk.VK_FALSE,
            lineWidth = 1.0
        )
        # Color blend state describes how blend factors are calculated (if used)
		# We need one blend attachment state per color attachment (even if blending is not used
        blendAttachmentState = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask = 0xf,
            blendEnable = vk.VK_FALSE
        )
        colorBlendState = vk.VkPipelineColorBlendStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount = 1,
            pAttachments = [blendAttachmentState]
        )
        # Viewport state sets the number of viewports and scissor used in this pipeline
		# Note: This is actually overriden by the dynamic states (see below)
        viewportState = vk.VkPipelineViewportStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount = 1,
            scissorCount = 1
        )
        # Enable dynamic states
        # Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
        #To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
        # For this example we will set the viewport and scissor using dynamic states
        dynamicStateEnables = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        dynamicState = vk.VkPipelineDynamicStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount = len(dynamicStateEnables),
            pDynamicStates = dynamicStateEnables
        )

        # Depth and stencil state containing depth and stencil compare and test operations
        # We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
        opState = vk.VkStencilOpState(
            failOp = vk.VK_STENCIL_OP_KEEP,
            passOp = vk.VK_STENCIL_OP_KEEP,
            compareOp = vk.VK_COMPARE_OP_ALWAYS
        )
        depthStencilState = vk.VkPipelineDepthStencilStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable = vk.VK_TRUE,
            depthWriteEnable = vk.VK_TRUE,
            depthCompareOp = vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            depthBoundsTestEnable = vk.VK_FALSE,
            stencilTestEnable = vk.VK_FALSE,
            front = opState,
            back = opState
        )
        #  Multi sampling state
        # This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
        multisampleState = vk.VkPipelineMultisampleStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples = vk.VK_SAMPLE_COUNT_1_BIT,
            pSampleMask = None
        )
        # Vertex input descriptions
        # Specifies the vertex input parameters for a pipeline
        #Vertex input binding
        # This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
        vertexInputBinding = vk.VkVertexInputBindingDescription(
            binding = 0,
            stride = self.vertexShape.size * self.vertexShape.itemsize,
            inputRate = vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        # Input attribute bindings describe shader attribute locations and memory layouts
        vertexInputAttributs = []
        # These match the following shader layout (see triangle.vert):
        #	layout (location = 0) in vec3 inPos;
        #	layout (location = 1) in vec3 inColor;
        # Attribute location 0: Position
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = 0,
            location = 0,
            # Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = 0 #  offsetof(vertexShape, position)
        )
        vertexInputAttributs.append(vertexInputAttribut)
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = 0,
            location = 1,
            # Color attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape[0].size * self.vertexShape.itemsize #  offsetof(vertexShape, color)
        )
        vertexInputAttributs.append(vertexInputAttribut)

        # Vertex input state used for pipeline creation
        vertexInputState = vk.VkPipelineVertexInputStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount = 1,
            pVertexBindingDescriptions = [vertexInputBinding],
            vertexAttributeDescriptionCount = len(vertexInputAttributs),
            pVertexAttributeDescriptions = vertexInputAttributs
        )
        # Shaders
        shaderStages = []
        # Vertex shader
        shaderStage = vk.VkPipelineShaderStageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            # Set pipeline stage for this shader
            stage = vk.VK_SHADER_STAGE_VERTEX_BIT,
            # Load binary SPIR-V shader
            module = vks.vulkantools.loadShader(self.getAssetPath() + "shaders/triangle/triangle.vert.spv", self.device),
            pName = "main"
        )
        shaderStages.append(shaderStage)
        # Fragment shader
        shaderStage = vk.VkPipelineShaderStageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            # Set pipeline stage for this shader
            stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            # Load binary SPIR-V shader
            module = vks.vulkantools.loadShader(self.getAssetPath() + "shaders/triangle/triangle.frag.spv", self.device),
            pName = "main"
        )
        shaderStages.append(shaderStage)

        # Assign the pipeline states to the pipeline creation info structure
        pipelineCreateInfo = vk.VkGraphicsPipelineCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            # The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
            layout = self.pipelineLayout,
            # Renderpass this pipeline is attached to
            renderPass = self.renderPass,
            pVertexInputState = vertexInputState,
            pInputAssemblyState = inputAssemblyState,
            pRasterizationState = rasterizationState,
            pColorBlendState = colorBlendState,
            pMultisampleState = multisampleState,
            pViewportState = viewportState,
            pDepthStencilState = depthStencilState,
            pDynamicState = dynamicState,
            stageCount = len(shaderStages),
            pStages = shaderStages
        )
        # Create rendering pipeline using the specified states
        self.pipelines = vk.vkCreateGraphicsPipelines(self.device, self.pipelineCache, 1, [pipelineCreateInfo], None)
        try:
            self.pipeline = self.pipelines[0]
        except TypeError:
            self.pipeline = self.pipelines
        # Shader modules are no longer needed once the graphics pipeline has been created
        vk.vkDestroyShaderModule(self.device, shaderStages[0].module, None)
        vk.vkDestroyShaderModule(self.device, shaderStages[1].module, None)

    def setupDescriptorPool(self):
        # We need to tell the API the number of max. requested descriptors per type
        typeCounts = []
        # This example only uses one descriptor type (uniform buffer) and only requests one descriptor of this type
        typeCount = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1
        )
        typeCounts.append(typeCount)

        # For additional types you need to add new entries in the type count list
        # E.g. for two combined image samplers :
        # typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        # typeCounts[1].descriptorCount = 2;

        # Create the global descriptor pool
        # All descriptors used in this example are allocated from this pool
        descriptorPoolInfo = vk.VkDescriptorPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext = None,
            poolSizeCount = 1,
            pPoolSizes = typeCounts,
            # Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
            maxSets = 1
        )
        self.descriptorPool = vk.vkCreateDescriptorPool(self.device, descriptorPoolInfo, None)

    def setupDescriptorSet(self):
        # Allocate a new descriptor set from the global descriptor pool
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = self.descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = [self.descriptorSetLayout]
        )
        self.descriptorSets = vk.vkAllocateDescriptorSets(self.device, allocInfo)
        self.descriptorSet = self.descriptorSets[0]

        # Update the descriptor set determining the shader binding points
        # For every binding point used in a shader there needs to be one
        # descriptor set matching that binding point
        writeDescriptorSet = vk.VkWriteDescriptorSet(
            # Binding 0 : Uniform buffer
            sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = self.descriptorSet,
            descriptorCount = 1,
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo = [self.uniformBufferVS['descriptor']],
            # Binds this uniform buffer to binding point 0
            dstBinding = 0
        )
        vk.vkUpdateDescriptorSets(self.device, 1, [writeDescriptorSet], 0, None)

    def buildCommandBuffers(self):
        """
Build separate command buffers for every framebuffer image
Unlike in OpenGL all rendering commands are recorded once into command buffers that are then resubmitted to the queue
This allows to generate work upfront and from multiple threads, one of the biggest advantages of Vulkan
        """
        cmdBufInfo = vk.VkCommandBufferBeginInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext = None
        )
        # Set clear values for all framebuffer attachments with loadOp set to clear
        # We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
        clearValues = []
        clearValue = vk.VkClearValue(
            color = [[ 0.0, 0.0, 0.2, 1.0 ]]
        )
        clearValues.append(clearValue)
        clearValue = vk.VkClearValue(
            depthStencil = [1.0, 0 ]
        )
        clearValues.append(clearValue)
        offset = vk.VkOffset2D(x = 0, y = 0)
        extent = vk.VkExtent2D(width = self.width, height = self.height)
        renderArea = vk.VkRect2D(offset = offset, extent = extent)
        for i in range(len(self.drawCmdBuffers)):
            renderPassBeginInfo = vk.VkRenderPassBeginInfo(
                sType = vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext = None,
                renderPass = self.renderPass,
                renderArea = renderArea,
                clearValueCount = 2,
                pClearValues = clearValues,
                # Set target frame buffer
                framebuffer = self.frameBuffers[i]
            )
            vk.vkBeginCommandBuffer(self.drawCmdBuffers[i], cmdBufInfo)
            # Start the first sub pass specified in our default render pass setup by the base class
            # This will clear the color and depth attachment
            vk.vkCmdBeginRenderPass(self.drawCmdBuffers[i], renderPassBeginInfo, vk.VK_SUBPASS_CONTENTS_INLINE)
            # Update dynamic viewport state
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

            # Bind descriptor sets describing shader binding points
            vk.vkCmdBindDescriptorSets(self.drawCmdBuffers[i], vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout, 0, 1, [self.descriptorSet], 0, None)
            # Bind the rendering pipeline
            # The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
            vk.vkCmdBindPipeline(self.drawCmdBuffers[i], vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline);
            # Bind triangle vertex buffer (contains position and colors)
            offsets = [ 0 ]
            vk.vkCmdBindVertexBuffers(self.drawCmdBuffers[i], 0, 1, [self.vertices['buffer']], offsets)
            # Bind triangle index buffer
            vk.vkCmdBindIndexBuffer(self.drawCmdBuffers[i], self.indices['buffer'], 0, vk.VK_INDEX_TYPE_UINT32)
            # Draw indexed triangle
            vk.vkCmdDrawIndexed(self.drawCmdBuffers[i], self.indices['count'], 1, 0, 0, 1)
            # uncomment for imgui support
            self.drawUI(self.drawCmdBuffers[i])
            vk.vkCmdEndRenderPass(self.drawCmdBuffers[i])
            # Ending the render pass will add an implicit barrier transitioning the frame buffer color attachment to
            # VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting it to the windowing system
            vk.vkEndCommandBuffer(self.drawCmdBuffers[i])

    def prepare(self):
        super().prepare()
        #self.prepareSynchronizationPrimitives()
        self.prepareVertices()
        self.prepareUniformBuffers()
        self.setupDescriptorSetLayout()
        self.preparePipelines()
        self.setupDescriptorPool()
        self.setupDescriptorSet()
        self.buildCommandBuffers()
        self.prepared = True

    def draw(self):
        nextBuffer=self.swapChain.acquireNextImage(self.semaphores['presentComplete'], self.currentBuffer)
        self.currentBuffer = nextBuffer
        #print(nextBuffer)
        # Use a fence to wait until the command buffer has finished execution before using it again
        vk.vkWaitForFences(self.device, 1, [self.waitFences[self.currentBuffer]], vk.VK_TRUE, vk.UINT64_MAX)
        vk.vkResetFences(self.device, 1, [self.waitFences[self.currentBuffer]])
        # Pipeline stage at which the queue submission will wait (via pWaitSemaphores)
        waitStageMask = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        # The submit info structure specifices a command buffer queue submission batch
        submitInfo = vk.VkSubmitInfo(
            sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pWaitDstStageMask = [ waitStageMask ],
            pWaitSemaphores = [ self.semaphores['presentComplete'] ],
            waitSemaphoreCount = 1,
            signalSemaphoreCount = 1,
            pSignalSemaphores = [ self.semaphores['renderComplete'] ],
            pCommandBuffers = [ self.drawCmdBuffers[self.currentBuffer] ],
            commandBufferCount = 1
        )
        # Submit to the graphics queue passing a wait fence
        vk.vkQueueSubmit(self.queue, 1, submitInfo, self.waitFences[self.currentBuffer])
        # Present the current buffer to the swap chain
        # Pass the semaphore signaled by the command buffer submission from the submit info as the wait semaphore for swap chain presentation
        # This ensures that the image is not presented to the windowing system until all commands have been submitted
        self.swapChain.queuePresent(self.queue, self.currentBuffer, self.semaphores['renderComplete'])


    def render(self):
        if not self.prepared:
            return
        #vk.vkDeviceWaitIdle(self.device)
        self.draw()
        # this one is required
        # uncomment for imgui support
        vk.vkDeviceWaitIdle(self.device)

    def viewChanged(self):
        self.updateUniformBuffers()

triangle = VulkanExample()
triangle.main()
