# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import imgui
import glm
import numpy as np

import vks.vulkantools
import vks.vulkanbuffer

class UIOverlay:
    def __init__(self):
        self.device = None
        self.queue = None
        self.rasterizationSamples = vk.VK_SAMPLE_COUNT_1_BIT
        self.subpass = 0
        self.vertexBuffer = vks.vulkanbuffer.Buffer()
        self.indexBuffer = vks.vulkanbuffer.Buffer()
        self.vertexCount = 0
        self.indexCount = 0
        self.shaders = []
        self.descriptorPool = None
        self.descriptorSetLayout = None
        self.descriptorSet = None
        self.pipelineLayout = None
        self.pipeline = None

        self.fontMemory = vk.VK_NULL_HANDLE
        self.fontImage = vk.VK_NULL_HANDLE
        self.fontView = vk.VK_NULL_HANDLE
        self.sampler = None
        self.pushConstBlock = {'scale': glm.vec2(), 'translate': glm.vec2()}
        self.pushConstBlockArray = np.array(glm.vec4(glm.vec2(), glm.vec2()))

        self.visible = True
        self.updated = False
        self.scale = 1.0

        imgui.create_context()
        #self.style = imgui.get_style() # don't know yet how to change self.style.color(imgui.COLOR_*)
        io = imgui.get_io()
        io.font_global_scale = self.scale

    def prepareResources(self):
        io = imgui.get_io()

        # Create font texture
        font = io.fonts.add_font_from_file_ttf("data/Roboto-Medium.ttf", 16.0)
        texWidth, texHeight, fontData = io.fonts.get_tex_data_as_rgba32()
        uploadSize = texWidth * texHeight * 4

        # Create target image for copy
        imageInfo = vk.VkImageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType = vk.VK_IMAGE_TYPE_2D,
            format = vk.VK_FORMAT_R8G8B8A8_UNORM,
            extent = (texWidth, texHeight, 1),
            mipLevels = 1,
            arrayLayers = 1,
            samples = vk.VK_SAMPLE_COUNT_1_BIT,
            tiling = vk.VK_IMAGE_TILING_OPTIMAL,
            usage = vk.VK_IMAGE_USAGE_SAMPLED_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self.fontImage = vk.vkCreateImage(self.device.logicalDevice, imageInfo, None)
        memReqs = vk.vkGetImageMemoryRequirements(self.device.logicalDevice, self.fontImage)
        memAlloc = vk.VkMemoryAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize = memReqs.size,
            memoryTypeIndex = self.device.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        )
        self.fontMemory = vk.vkAllocateMemory(self.device.logicalDevice, memAlloc, None)
        vk.vkBindImageMemory(self.device.logicalDevice, self.fontImage, self.fontMemory, 0)

        # Image view
        aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT
        subresourceRange = vk.VkImageSubresourceRange(
            levelCount = 1,
            layerCount = 1,
            aspectMask = aspectMask
        )
        imageViewCI = vk.VkImageViewCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            viewType = vk.VK_IMAGE_VIEW_TYPE_2D,
            image = self.fontImage,
            format = vk.VK_FORMAT_R8G8B8A8_UNORM,
            subresourceRange = subresourceRange
        )
        self.fontView = vk.vkCreateImageView(self.device.logicalDevice, imageViewCI, None)

        # Staging buffers for font data upload
        stagingBuffer = self.device.createvksBuffer(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uploadSize)
        stagingBuffer.map()
        stagingBuffer.copyTo(fontData, uploadSize)
        stagingBuffer.unmap()
        # Copy buffer data to font image
        copyCmd = self.device.createCommandBuffer(vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, True)
        #  Prepare for transfer
        vks.vulkantools.setImageLayout(copyCmd, self.fontImage, vk.VK_IMAGE_ASPECT_COLOR_BIT,
            vk.VK_IMAGE_LAYOUT_UNDEFINED,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.VK_PIPELINE_STAGE_HOST_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT)
        # Copy
        imageSubresource = vk.VkImageSubresourceLayers(aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount = 1)
        imageExtent = vk.VkExtent3D(width = texWidth, height = texHeight, depth = 1)
        bufferCopyRegion = vk.VkBufferImageCopy(
            imageSubresource = imageSubresource,
            imageExtent = imageExtent
        )
        vk.vkCmdCopyBufferToImage(copyCmd, stagingBuffer.buffer, self.fontImage, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, bufferCopyRegion)

        # Prepare for shader read
        vks.vulkantools.setImageLayout(copyCmd, self.fontImage, vk.VK_IMAGE_ASPECT_COLOR_BIT,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)

        self.device.flushCommandBuffer(copyCmd, self.queue, True)
        stagingBuffer.destroy()

        # Font texture Sampler
        samplerInfo = vk.VkSamplerCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            maxAnisotropy = 1.0,
            magFilter = vk.VK_FILTER_LINEAR,
            minFilter = vk.VK_FILTER_LINEAR,
            mipmapMode = vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
            addressModeU = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            borderColor = vk.VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
        )
        self.sampler = vk.vkCreateSampler(self.device.logicalDevice, samplerInfo, None)
        # Descriptor pool
        poolSizes = []
        poolSizes.append(vk.VkDescriptorPoolSize(type = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount = 1))
        descriptorPoolInfo = vk.VkDescriptorPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = len(poolSizes),
            pPoolSizes = poolSizes,
            maxSets = 2
        )
        self.descriptorPool = vk.vkCreateDescriptorPool(self.device.logicalDevice, descriptorPoolInfo, None)
        #Descriptor set layout
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
        self.descriptorSetLayout = vk.vkCreateDescriptorSetLayout(self.device.logicalDevice, descriptorLayout, None)
        # Descriptor set
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = self.descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = [self.descriptorSetLayout]
        )
        self.descriptorSet = vk.vkAllocateDescriptorSets(self.device.logicalDevice, allocInfo)[0]
        fontDescriptor = vk.VkDescriptorImageInfo(
            sampler = self.sampler,
            imageView = self.fontView,
            imageLayout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        )
        writeDescriptorSets = []
        writeDescriptorSet = vk.VkWriteDescriptorSet(
            sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = self.descriptorSet,
            descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            dstBinding = 0,
            pImageInfo = [ fontDescriptor ],
            descriptorCount = 1
        )
        writeDescriptorSets.append(writeDescriptorSet)
        vk.vkUpdateDescriptorSets(self.device.logicalDevice, len(writeDescriptorSets), writeDescriptorSets, 0, None)

    def preparePipeline(self, pipelineCache, renderPass):
        """
Prepare a separate pipeline for the UI overlay rendering decoupled from the main application
        """
        pushConstantRange = vk.VkPushConstantRange(
            stageFlags = vk.VK_SHADER_STAGE_VERTEX_BIT,
            offset = 0,
            size = self.pushConstBlockArray.size * self.pushConstBlockArray.itemsize
        )
        pipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = 1,
            pSetLayouts = [self.descriptorSetLayout],
            pushConstantRangeCount = 1,
            pPushConstantRanges = [ pushConstantRange ]
        )
        self.pipelineLayout = vk.vkCreatePipelineLayout(self.device.logicalDevice, pipelineLayoutCreateInfo, None)
        inputAssemblyState = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology = vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            flags = 0,
            primitiveRestartEnable = vk.VK_FALSE
        )
        rasterizationState = vk.VkPipelineRasterizationStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = vk.VK_POLYGON_MODE_FILL,
            cullMode = vk.VK_CULL_MODE_NONE,
            frontFace = vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthClampEnable = vk.VK_FALSE,
            rasterizerDiscardEnable = vk.VK_FALSE,
            depthBiasEnable = vk.VK_FALSE,
            lineWidth = 1.0,
            flags = 0
        )
        blendAttachmentState = vk.VkPipelineColorBlendAttachmentState(
            blendEnable = vk.VK_TRUE,
            colorWriteMask = vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT,
            srcColorBlendFactor = vk.VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor = vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp = vk.VK_BLEND_OP_ADD,
            srcAlphaBlendFactor = vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            dstAlphaBlendFactor = vk.VK_BLEND_FACTOR_ZERO,
            alphaBlendOp = vk.VK_BLEND_OP_ADD
        )
        colorBlendState = vk.VkPipelineColorBlendStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount = 1,
            pAttachments = [blendAttachmentState]
        )
        opState = vk.VkStencilOpState(
            failOp = vk.VK_STENCIL_OP_KEEP,
            passOp = vk.VK_STENCIL_OP_KEEP,
            compareOp = vk.VK_COMPARE_OP_ALWAYS
        )
        depthStencilState = vk.VkPipelineDepthStencilStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable = vk.VK_FALSE,
            depthWriteEnable = vk.VK_FALSE,
            depthCompareOp = vk.VK_COMPARE_OP_ALWAYS,
            depthBoundsTestEnable = vk.VK_FALSE,
            stencilTestEnable = vk.VK_FALSE,
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
            rasterizationSamples = self.rasterizationSamples,
            flags = 0
        )
        dynamicStateEnables = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        dynamicState = vk.VkPipelineDynamicStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount = len(dynamicStateEnables),
            pDynamicStates = dynamicStateEnables,
            flags = 0
        )
        vertexInputBinding = vk.VkVertexInputBindingDescription(
            binding = 0,
            stride = imgui.vertex_buffer_vertex_size(),
            inputRate = vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        vertexInputAttributes = []
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = 0,
            location = 0,
            format = vk.VK_FORMAT_R32G32_SFLOAT,
            offset =  imgui.vertex_buffer_vertex_pos_offset()
        )
        vertexInputAttributes.append(vertexInputAttribut)
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = 0,
            location = 1,
            format = vk.VK_FORMAT_R32G32_SFLOAT,
            offset =  imgui.vertex_buffer_vertex_uv_offset()
        )
        vertexInputAttributes.append(vertexInputAttribut)
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = 0,
            location = 2,
            format = vk.VK_FORMAT_R8G8B8A8_UNORM,
            offset =  imgui.vertex_buffer_vertex_col_offset()
        )
        vertexInputAttributes.append(vertexInputAttribut)
        vertexInputState = vk.VkPipelineVertexInputStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount = 1,
            pVertexBindingDescriptions = [vertexInputBinding],
            vertexAttributeDescriptionCount = len(vertexInputAttributes),
            pVertexAttributeDescriptions = vertexInputAttributes
        )
        pipelineCreateInfo = vk.VkGraphicsPipelineCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            layout = self.pipelineLayout,
            renderPass = renderPass,
            flags = 0,
            basePipelineIndex = -1,
            basePipelineHandle = vk.VK_NULL_HANDLE,
            pVertexInputState = vertexInputState,
            pInputAssemblyState = inputAssemblyState,
            pRasterizationState = rasterizationState,
            pColorBlendState = colorBlendState,
            pMultisampleState = multisampleState,
            pViewportState = viewportState,
            pDepthStencilState = depthStencilState,
            pDynamicState = dynamicState,
            stageCount = len(self.shaders),
            pStages = self.shaders,
            subpass = self.subpass,
        )
        # Create rendering pipeline using the specified states
        self.pipelines = vk.vkCreateGraphicsPipelines(self.device.logicalDevice, pipelineCache, 1, [pipelineCreateInfo], None)
        try:
            self.pipeline = self.pipelines[0]
        except TypeError:
            self.pipeline = self.pipelines

    def update(self):
        imDrawData = imgui.get_draw_data()
        updateCmdBuffers = False
        if imDrawData is None:
            return False
        # Note: Alignment is done inside buffer creation
        vertexBufferSize = imDrawData.total_vtx_count * imgui.vertex_buffer_vertex_size()
        indexBufferSize = imDrawData.total_idx_count * imgui.index_buffer_index_size()
        if vertexBufferSize == 0 or indexBufferSize == 0:
            return False
        # Vertex buffer
        if self.vertexBuffer.buffer == vk.VK_NULL_HANDLE or self.vertexCount != imDrawData.total_vtx_count:
            self.vertexBuffer.unmap()
            self.vertexBuffer.destroy()
            self.vertexBuffer = self.device.createvksBuffer(vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, vertexBufferSize)
            self.vertexCount = imDrawData.total_vtx_count
            self.vertexBuffer.unmap()
            self.vertexBuffer.map()
            updateCmdBuffers = True
        # Index buffer
        if self.indexBuffer.buffer == vk.VK_NULL_HANDLE or self.indexCount != imDrawData.total_idx_count:
            self.indexBuffer.unmap()
            self.indexBuffer.destroy()
            self.indexBuffer = self.device.createvksBuffer(vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, indexBufferSize)
            self.indexCount = imDrawData.total_idx_count
            self.indexBuffer.unmap()
            self.indexBuffer.map()
            updateCmdBuffers = True
        #print(vertexBufferSize, self.vertexBuffer.size)
        # TODO allocate whole array (sum([size(buf) for buf in commands_lists]) or total_vtx_count?)
        # and make region subcopies as these are C pointers (imgui -> vulkan)
        # vertexdatawrapper = vks.vulkanbuffer.DataWrapper()
        # vertexarray = np.empty(shape = (0,), dtype = np.uint8)
        # indexdatawrapper = vks.vulkanbuffer.DataWrapper()
        # indexarray = np.empty(shape = (0,), dtype = np.uint8)
        # for i in range(len(imDrawData.commands_lists)):
        #     cmd_list = imDrawData.commands_lists[i]
        #     vertexdatawrapper.__array_interface__['shape'] = (cmd_list.vtx_buffer_size * imgui.vertex_buffer_vertex_size(),)
        #     vertexdatawrapper.__array_interface__['data'] = (cmd_list.vtx_buffer_data, False)
        #     # TODO use np.copyto with a subarray as destination
        #     vertexarray = np.append(vertexarray, vertexdatawrapper)
        #     indexdatawrapper.__array_interface__['shape'] = (cmd_list.idx_buffer_size * imgui.index_buffer_index_size(),)
        #     indexdatawrapper.__array_interface__['data'] = (cmd_list.idx_buffer_data, False)
        #     # TODO use np.copyto with a subarray as destination
        #     indexarray = np.append(indexarray, indexdatawrapper)
        # vertexarray = np.append(vertexarray, np.zeros(shape=(self.vertexBuffer.size - vertexBufferSize,), dtype=np.uint8))
        # vertexbufferwrapper = np.array(self.vertexBuffer.mapped, copy=False)
        # np.copyto(vertexbufferwrapper, vertexarray, casting='no')
        # indexarray = np.append(indexarray, np.zeros(shape=(self.indexBuffer.size - indexBufferSize,), dtype=np.uint8))
        # indexbufferwrapper = np.array(self.indexBuffer.mapped, copy=False)
        # np.copyto(indexbufferwrapper, indexarray, casting='no')

        vertexdatawrapper = vks.vulkanbuffer.DataWrapper()
        # vertexbufferwrapper = vks.vulkanbuffer.DataWrapper()
        vertexbufferwrapper = np.array(self.vertexBuffer.mapped, copy=False)
        vertexbufferindex = 0
        indexdatawrapper = vks.vulkanbuffer.DataWrapper()
        # indexbufferwrapper = vks.vulkanbuffer.DataWrapper()
        indexbufferwrapper = np.array(self.indexBuffer.mapped, copy=False)
        indexbufferindex = 0
        # indexarray = np.empty(shape = (0,), dtype = np.uint8)
        for i in range(len(imDrawData.commands_lists)):
            cmd_list = imDrawData.commands_lists[i]
            vertexdatawrapper.__array_interface__['shape'] = (cmd_list.vtx_buffer_size * imgui.vertex_buffer_vertex_size(),)
            vertexdatawrapper.__array_interface__['data'] = (cmd_list.vtx_buffer_data, False)
            #vertexbufferwrapper.__array_interface__['shape'] = (cmd_list.vtx_buffer_size * imgui.vertex_buffer_vertex_size(),)
            #vertexbufferwrapper.__array_interface__['data'] = (self.vertexBuffer.mapped[vertexbufferindex], True)
            np.copyto(vertexbufferwrapper[vertexbufferindex:vertexbufferindex+(cmd_list.vtx_buffer_size * imgui.vertex_buffer_vertex_size())], vertexdatawrapper, casting='no')
            vertexbufferindex += cmd_list.vtx_buffer_size * imgui.vertex_buffer_vertex_size()

            indexdatawrapper.__array_interface__['shape'] = (cmd_list.idx_buffer_size * imgui.index_buffer_index_size(),)
            indexdatawrapper.__array_interface__['data'] = (cmd_list.idx_buffer_data, False)
            #indexbufferwrapper.__array_interface__['shape'] = (cmd_list.idx_buffer_size * imgui.index_buffer_index_size(),)
            #indexbufferwrapper.__array_interface__['data'] = (self.indexBuffer.mapped + indexbufferindex, True)
            np.copyto(indexbufferwrapper[indexbufferindex:indexbufferindex+(cmd_list.idx_buffer_size * imgui.index_buffer_index_size())], indexdatawrapper, casting='no')
            indexbufferindex += cmd_list.idx_buffer_size * imgui.index_buffer_index_size()

        self.vertexBuffer.flush()
        self.indexBuffer.flush()
        return updateCmdBuffers

    def draw(self, commandBuffer):
        imDrawData = imgui.get_draw_data()
        vertexOffset = 0
        indexOffset = 0
        if imDrawData is None or imDrawData.cmd_count == 0:
            return
        io = imgui.get_io()
        vk.vkCmdBindPipeline(commandBuffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)
        vk.vkCmdBindDescriptorSets(commandBuffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout, 0, 1, [self.descriptorSet], 0, None)
        self.pushConstBlock['scale'] = glm.vec2(2.0 / io.display_size.x, 2.0 / io.display_size.y)
        self.pushConstBlock['translate'] = glm.vec2(-1.0)
        self.pushConstBlockArray = np.array(glm.vec4(self.pushConstBlock['scale'], self.pushConstBlock['translate']))
        pushConstBlockArraySize = self.pushConstBlockArray.size * self.pushConstBlockArray.itemsize
        #print(pushConstBlockArraySize, self.pushConstBlockArray, self.pipelineLayout, commandBuffer)
        #from vulkan._vulkancache import ffi
        #print(type(self.pushConstBlockArray.__array_interface__['data'][0]))
        #pBlockArray = ffi.cast('void *', self.pushConstBlockArray.__array_interface__['data'][0])
        pBlockArray = self.pushConstBlockArray.__array_interface__['data'][0]
        #vk.vkCmdPushConstants(commandBuffer, self.pipelineLayout, vk.VK_SHADER_STAGE_VERTEX_BIT, 0, pushConstBlockArraySize, self.pushConstBlockArray)
        vk.vkCmdPushConstants(commandBuffer, self.pipelineLayout, vk.VK_SHADER_STAGE_VERTEX_BIT, 0, pushConstBlockArraySize, pBlockArray)

        offsets = [0]
        vk.vkCmdBindVertexBuffers(commandBuffer, 0, 1, [self.vertexBuffer.buffer], offsets)
        vk.vkCmdBindIndexBuffer(commandBuffer, self.indexBuffer.buffer, 0, vk.VK_INDEX_TYPE_UINT16)

        for i in range(len(imDrawData.commands_lists)):
            cmd_list = imDrawData.commands_lists[i]
            for j in range(len(cmd_list.commands)):
                pcmd = cmd_list.commands[j]
                cmdoffset = vk.VkOffset2D(x = max(int(pcmd.clip_rect.x), 0), y = max(int(pcmd.clip_rect.y), 0))
                cmdextent = vk.VkExtent2D(width = int(pcmd.clip_rect.z - pcmd.clip_rect.x), height = int(pcmd.clip_rect.w - pcmd.clip_rect.y))
                scissorRect = vk.VkRect2D(offset = cmdoffset, extent = cmdextent)
                vk.vkCmdSetScissor(commandBuffer, 0, 1, scissorRect)
                vk.vkCmdDrawIndexed(commandBuffer, pcmd.elem_count, 1, indexOffset, vertexOffset, 0)
                indexOffset += pcmd.elem_count
            vertexOffset += cmd_list.vtx_buffer_size

    def freeResources(self):
        imgui.destroy_context()
        self.vertexBuffer.destroy()
        self.indexBuffer.destroy()
        vk.vkDestroyImageView(self.device.logicalDevice, self.fontView, None)
        vk.vkDestroyImage(self.device.logicalDevice, self.fontImage, None)
        vk.vkFreeMemory(self.device.logicalDevice, self.fontMemory, None)
        vk.vkDestroySampler(self.device.logicalDevice, self.sampler, None)
        vk.vkDestroyDescriptorSetLayout(self.device.logicalDevice, self.descriptorSetLayout, None)
        vk.vkDestroyDescriptorPool(self.device.logicalDevice, self.descriptorPool, None)
        vk.vkDestroyPipelineLayout(self.device.logicalDevice, self.pipelineLayout, None)
        vk.vkDestroyPipeline(self.device.logicalDevice, self.pipeline, None)
