# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk

import vks.vulkanbuffer

class VulkanDevice:
    def __init__(self, physicalDevice):
        assert(physicalDevice is not None)
        self.physicalDevice = physicalDevice
        self.logicalDevice = None
        # Store Properties features, limits and properties of the physical device for later use
        # Device properties also contain limits and sparse properties
        self.properties = vk.vkGetPhysicalDeviceProperties(self.physicalDevice)
        # Features should be checked by the examples before using them
        self.features = vk.vkGetPhysicalDeviceFeatures(self.physicalDevice)
        self.enabledFeatures = None
        # Memory properties are used regularly for creating all kinds of buffers
        self.memoryProperties = vk.vkGetPhysicalDeviceMemoryProperties(self.physicalDevice)
        # Queue family properties, used for setting up requested queues upon device creation
        self.queueFamilyProperties = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physicalDevice)
        # Get list of supported extensions
        extensions = vk.vkEnumerateDeviceExtensionProperties(self.physicalDevice, None)
        self.supportedExtensions = []
        for e in extensions:
            self.supportedExtensions.append(e.extensionName)
        self.commandPool = vk.VK_NULL_HANDLE
        self.enableDebugMarkers = False
        self.queueFamilyIndices = { 'graphics': None, 'compute':None, 'transfer':None}
    def cleanup(self):
        if self.commandPool:
            vk.vkDestroyCommandPool(self.logicalDevice, self.commandPool, None)
        if self.logicalDevice:
            vk.vkDestroyDevice(self.logicalDevice, None)

    def getMemoryType(self, typeBits, properties):
        for i in range(self.memoryProperties.memoryTypeCount):
            if (typeBits & 1) == 1:
                if (self.memoryProperties.memoryTypes[i].propertyFlags & properties) == properties:
                    return i
            typeBits >>= 1
        raise RuntimeError('Could not find a matching memory type')

    def getQueueFamilyIndex(self, queueFlags):
        # Dedicated queue for compute
		# Try to find a queue family index that supports compute but not graphics
        if (queueFlags & vk.VK_QUEUE_COMPUTE_BIT):
            for i in range(len(self.queueFamilyProperties)):
                if (self.queueFamilyProperties[i].queueFlags & queueFlags) and ((self.queueFamilyProperties[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT) == 0):
                    return i
        # Dedicated queue for transfer
	    # Try to find a queue family index that supports transfer but not graphics and compute
        if (queueFlags & vk.VK_QUEUE_TRANSFER_BIT):
            for i in range(len(self.queueFamilyProperties)):
                if (self.queueFamilyProperties[i].queueFlags & queueFlags) and ((self.queueFamilyProperties[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT) == 0) and ((self.queueFamilyProperties[i].queueFlags & vk.VK_QUEUE_COMPUTE_BIT) == 0):
                    return i
        #  For other queue types or if no separate compute queue is present, return the first one to support the requested flags
        for i in range(len(self.queueFamilyProperties)):
            if (self.queueFamilyProperties[i].queueFlags & queueFlags):
                return i
        raise RuntimeError('Could not find a matching queue family index')

    def createLogicalDevice(self, enabledFeatures, enabledExtensions, useSwapChain=True, requestedQueueTypes = vk.VK_QUEUE_GRAPHICS_BIT | vk.VK_QUEUE_COMPUTE_BIT):
        queueCreateInfos = []
        defaultQueuePriority = 0.0
        # Graphics queue
        if (requestedQueueTypes & vk.VK_QUEUE_GRAPHICS_BIT):
            self.queueFamilyIndices['graphics'] = self.getQueueFamilyIndex(vk.VK_QUEUE_GRAPHICS_BIT)
            queueInfo = vk.VkDeviceQueueCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex = self.queueFamilyIndices['graphics'],
                queueCount = 1,
                pQueuePriorities = [defaultQueuePriority])
            queueCreateInfos.append(queueInfo)
        else:
            queueFamilyIndices['graphics'] = vk.VK_NULL_HANDLE
        # Dedicated compute queue
        if (requestedQueueTypes & vk.VK_QUEUE_COMPUTE_BIT):
            self.queueFamilyIndices['compute'] = self.getQueueFamilyIndex(vk.VK_QUEUE_COMPUTE_BIT)
            if self.queueFamilyIndices['compute'] != self.queueFamilyIndices['graphics']:
                # If compute family index differs, we need an additional queue create info for the compute queue
                queueInfo = vk.VkDeviceQueueCreateInfo(
                    sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = self.queueFamilyIndices['compute'],
                    queueCount = 1,
                    pQueuePriorities = [defaultQueuePriority])
                queueCreateInfos.append(queueInfo)
        else:
            self.queueFamilyIndices['compute'] = self.queueFamilyIndices['graphics']
        # Dedicated transfer queue
        if (requestedQueueTypes & vk.VK_QUEUE_TRANSFER_BIT):
            self.queueFamilyIndices['transfer'] = self.getQueueFamilyIndex(vk.VK_QUEUE_TRANSFER_BIT)
            if (self.queueFamilyIndices['transfer'] != self.queueFamilyIndices['graphics']) and (self.queueFamilyIndices['transfer'] != self.queueFamilyIndices['compute']):
                # If transfer family index differs, we need an additional queue create info for the compute queue
                queueInfo = vk.VkDeviceQueueCreateInfo(
                    sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = self.queueFamilyIndices['transfer'],
                    queueCount = 1,
                    pQueuePriorities = [defaultQueuePriority])
                queueCreateInfos.append(queueInfo)
        else:
            self.queueFamilyIndices['transfer'] = self.queueFamilyIndices['graphics']

        self.enabledFeatures = enabledFeatures

        deviceExtensions = enabledExtensions.copy()
        if (useSwapChain):
            # If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
            deviceExtensions.append(vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME)
        #deviceCreateInfo = vk.VkDeviceCreateInfo(
        #    sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        #    pQueueCreateInfos = queueCreateInfos,
        #    pEnabledFeatures = enabledFeatures)
        if self.extensionSupported(vk.VK_EXT_DEBUG_MARKER_EXTENSION_NAME):
            deviceExtensions.append(vk.VK_EXT_DEBUG_MARKER_EXTENSION_NAME)
            self.enableDebugMarkers = True
        #if len(deviceExtensions) > 0:
        #    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions
        deviceCreateInfo = vk.VkDeviceCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pQueueCreateInfos = queueCreateInfos,
            pEnabledFeatures = enabledFeatures,
            ppEnabledExtensionNames = deviceExtensions)
        try:
            self.logicalDevice = vk.vkCreateDevice(self.physicalDevice, deviceCreateInfo, pAllocator=None)
        except Exception as e:
            raise e
        # Create a default command pool for graphics command buffers
        self.commandPool = self.createCommandPool(self.queueFamilyIndices['graphics'])
        return

    def createCommandPool(self, queueFamilyIndex, createFlags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT):
        """
    Create a command pool for allocation command buffers from
	*
	* @param queueFamilyIndex Family index of the queue to create the command pool for
	* @param createFlags (Optional) Command pool creation flags (Defaults to VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
	*
	* @note Command buffers allocated from the created pool can only be submitted to a queue with the same family index
	*
	* @return A handle to the created command buffer
        """
        cmdPoolInfo = vk.VkCommandPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex = queueFamilyIndex,
            flags = createFlags)
        return vk.vkCreateCommandPool(self.logicalDevice, cmdPoolInfo, pAllocator = None)

    def createCommandBuffer(self, level, begin = False):
        """
Allocate a command buffer from the command pool

@param level Level of the new command buffer (primary or secondary)
@param (Optional) begin If true, recording on the new command buffer will be started (vkBeginCommandBuffer) (Defaults to false)

@return A handle to the allocated command buffer
        """
        cmdBufAllocateInfo = vk.VkCommandBufferAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool = self.commandPool,
            level = level,
            commandBufferCount = 1
        )
        cmdBuffer = vk.vkAllocateCommandBuffers(self.logicalDevice, cmdBufAllocateInfo)[0]

        if begin:
            cmdBufInfo = vk.VkCommandBufferBeginInfo(sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
            vk.vkBeginCommandBuffer(cmdBuffer, cmdBufInfo)
        return cmdBuffer

    def flushCommandBuffer(self, commandBuffer, queue, free = True):
        """
Finish command buffer recording and submit it to a queue

@param commandBuffer Command buffer to flush
@param queue Queue to submit the command buffer to
@param free (Optional) Free the command buffer once it has been submitted (Defaults to true)
@note The queue that the command buffer is submitted to must be from the same family index as the pool it was allocated from
@note Uses a fence to ensure command buffer has finished executing
        """
        if commandBuffer == vk.VK_NULL_HANDLE:
            return
        vk.vkEndCommandBuffer(commandBuffer)
        submitInfo = vk.VkSubmitInfo(
            sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = [commandBuffer]
        )
        # Create fence to ensure that the command buffer has finished executing
        fenceCreateInfo = vk.VkFenceCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags = 0 # vk.VK_FLAGS_NONE
        )
        fence = vk.vkCreateFence(self.logicalDevice, fenceCreateInfo, None)
        # Submit to the queue
        vk.vkQueueSubmit(queue, 1, [submitInfo], fence)
        # Wait for the fence to signal that command buffer has finished executing
        vk.vkWaitForFences(self.logicalDevice, 1, [fence], vk.VK_TRUE, vks.vulkanglobals.DEFAULT_FENCE_TIMEOUT)

        vk.vkDestroyFence(self.logicalDevice, fence, None)
        if free:
            vk.vkFreeCommandBuffers(self.logicalDevice, self.commandPool, 1, [commandBuffer])

    def extensionSupported(self, extension):
        return extension in self.supportedExtensions

    def createBuffer(self, usageFlags, memoryPropertyFlags, size, data):
        """
Create a buffer on the device

@param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
@param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
@param size Size of the buffer in byes
@return buffer Pointer to the buffer handle acquired by the function
@return memory Pointer to the memory handle acquired by the function
@param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)

@return (buffer, memory)
        """
        raise NotImplementedError

    def createvksBuffer(self, usageFlags, memoryPropertyFlags, size, data = None):
        """
Create a vks.buffer on the device

@param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
@param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
@param size Size of the buffer in byes
@return vks.buffer Pointer to the vks.buffer handle acquired by the function
@return memory Pointer to the memory handle acquired by the function
@param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)

@return vks.buffer
        """
        buffer = vks.vulkanbuffer.Buffer()
        buffer.device = self.logicalDevice
        bufferCreateInfo = vk.VkBufferCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size = size,
            usage = usageFlags
        )
        buffer.buffer = vk.vkCreateBuffer(self.logicalDevice, bufferCreateInfo, None)

        memReqs = vk.vkGetBufferMemoryRequirements(self.logicalDevice, buffer.buffer)
        memAlloc = vk.VkMemoryAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize = memReqs.size,
            memoryTypeIndex = self.getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags)
        )
        buffer.memory = vk.vkAllocateMemory(self.logicalDevice, memAlloc, None)

        buffer.alignment = memReqs.alignment
        buffer.size = memAlloc.allocationSize
        buffer.usageFlags = usageFlags
        buffer.memoryPropertyFlags = memoryPropertyFlags

        if data is not None:
            buffer.map()
            buffer.copyTo(data, size) # TODO: check allocationSize also
            if memoryPropertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT == 0:
                buffer.flush()
            buffer.unmap()

        # Initialize a default descriptor that covers the whole buffer size
        buffer.setupDescriptor()
        #Attach the memory to the buffer object
        buffer.bind()
        return buffer
