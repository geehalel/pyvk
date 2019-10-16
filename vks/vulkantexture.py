# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk

import vks.vulkanbuffer
import vks.vulkandevice
import vks.ktxfile

import numpy as np

class Texture:
    """
Vulkan texture base class
    """
    def __init__(self):
        self.device = None
        self.image = None
        self.imageLayout = None
        self.deviceMemory = None
        self.view = None
        self.width = 0
        self.height = 0
        self.mipLevels = 0
        self.layerCount = 0
        self.descriptor = None
        # Optional sampler to use with this texture
        self.sampler = None

    def updateDescriptor(self):
        """
Update image descriptor from current sampler, view and image layout
        """
        self.descriptor.sampler = self.sampler
        self.descriptor.imageView = self.imageView
        self.descriptor.imageLayout = self.imageLayout

    def destroy(self):
        """
Release all Vulkan resources held by this texture
        """
        vk.vkDestroyImageView(self.device.logicalDevice, self.view, None)
        vk.vkDestroyImage(self.device.logicalDevice, self.image, None)
        if self.sampler is not None:
            vk.vkDestroySampler(self.device.logicalDevice, self.sampler, None)
        vk.vkFreeMemory(self.device.logicalDevice, self.deviceMemory, None)

class Texture2D(Texture):
    """
2D texture
    """
    def __init__(self):
        super().__init__()

    def loadFromFile(self, filename, format, device, copyQueue,
        imageUsageFlags = vk.VK_IMAGE_USAGE_SAMPLED_BIT,
        imageLayout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        forceLinear = False):
        """
Load a 2D texture including all mip levels
* @param filename File to load (supports .ktx and .dds)
* @param format Vulkan format of the image data stored in the file
* @param device Vulkan device to create the texture on
* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
* @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
* @param (Optional) forceLinear Force linear tiling (not advised, defaults to false	)
        """
        self.device = device
        texfile = vks.ktxfile.KtxFile(filename, "r")
        self.width = texfile.ktx_pixelwidth
        self.height = texfile.ktx_pixelheight
        self.mipLevels = texfile.ktx_nummimpmaplevels
        formatProperties = vk.vkGetPhysicalDeviceFormatProperties(self.device.physicalDevice, format)

        # Only use linear tiling if requested (and supported by the device)
        # Support for linear tiling is mostly limited, so prefer to use
        # optimal tiling instead
        # On most implementations linear tiling will only support a very
        # limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
        useStaging = not forceLinear
        print(texfile.size)
        if useStaging:
            bufferCreateInfo = vk.VkBufferCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = texfile.size,
                usage = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE
            )
            stagingBuffer = vk.vkCreateBuffer(self.device.logicalDevice, bufferCreateInfo, None)
            memReqs = vk.vkGetBufferMemoryRequirements(self.device.logicalDevice, stagingBuffer)
            memAlloc = vk.VkMemoryAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = self.device.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            )
            stagingMemory = vk.vkAllocateMemory(self.device.logicalDevice, memAlloc, None)
            vk.vkBindBufferMemory(self.device.logicalDevice, stagingBuffer, stagingMemory, 0)
            data = vk.vkMapMemory(self.device.logicalDevice, stagingMemory, 0, memAlloc.allocationSize, 0)
            datawrapper = np.array(data, copy=False)
            texdata = b''
            for i in range(texfile.ktx_nummimpmaplevels):
                texdata += texfile.get_image(i)
            texbuf = np.resize(texdata, memReqs.size)
            np.copyto(datawrapper, texdata, casting='no')
            vkUnmapMemory(self.device.logicalDevice, stagingMemory)



        else:
            raise(NotImplementedError, 'Non staging buffers are not implemented yet')
