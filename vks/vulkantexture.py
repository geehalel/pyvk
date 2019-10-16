# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk

import vks.vulkanbuffer
import vks.vulkandevice
import vks.vulkantools
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
        self.descriptor = vk.VkDescriptorImageInfo()
        # Optional sampler to use with this texture
        self.sampler = None

    def updateDescriptor(self):
        """
Update image descriptor from current sampler, view and image layout
        """
        self.descriptor.sampler = self.sampler
        self.descriptor.imageView = self.view
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

        copyCmd = self.device.createCommandBuffer(vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, True)

        # Only use linear tiling if requested (and supported by the device)
        # Support for linear tiling is mostly limited, so prefer to use
        # optimal tiling instead
        # On most implementations linear tiling will only support a very
        # limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
        useStaging = not forceLinear
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
            if texfile.size < memReqs.size:
                texdata += (texfile.size - memReqs.size) * b'\x00'
            texbuf = vks.vulkanbuffer.DataWrapper()
            texbuf.__array_interface__['shape'] = (memReqs.size,)
            texbuf.__array_interface__['data'] = texdata
            np.copyto(datawrapper, texbuf, casting='no')
            vk.vkUnmapMemory(self.device.logicalDevice, stagingMemory)

            bufferCopyRegions = []
            offset = 0
            for i in range(self.mipLevels):
                imageSubresource = vk.VkImageSubresourceLayers(aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount = 1, mipLevel = i, baseArrayLayer = 0)
                imageExtent = vk.VkExtent3D(width = texfile.ktx_images[i]['width'], height = texfile.ktx_images[i]['height'], depth = 1)
                bufferCopyRegion = vk.VkBufferImageCopy(
                    imageSubresource = imageSubresource,
                    imageExtent = imageExtent,
                    bufferOffset = offset
                )
                bufferCopyRegions.append(bufferCopyRegion)
                offset += texfile.ktx_images[i]['size']
            # Create optimal tiled target image
            imageCreateInfo = vk.VkImageCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType = vk.VK_IMAGE_TYPE_2D,
                format = format,
                extent = (self.width, self.height, 1),
                mipLevels = self.mipLevels,
                arrayLayers = 1,
                samples = vk.VK_SAMPLE_COUNT_1_BIT,
                tiling = vk.VK_IMAGE_TILING_OPTIMAL,
                # Ensure that the TRANSFER_DST bit is set for staging
                usage = imageUsageFlags | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED
            )
            self.image = vk.vkCreateImage(self.device.logicalDevice, imageCreateInfo, None)
            memReqs = vk.vkGetImageMemoryRequirements(self.device.logicalDevice, self.image)
            memAlloc = vk.VkMemoryAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = self.device.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            )
            self.deviceMemory = vk.vkAllocateMemory(self.device.logicalDevice, memAlloc, None)
            vk.vkBindImageMemory(self.device.logicalDevice, self.image, self.deviceMemory, 0)
            subresourceRange = vk.VkImageSubresourceRange(
                levelCount = self.mipLevels,
                baseMipLevel = 0,
                layerCount = 1,
                aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT
            )
            vks.vulkantools.setImageLayoutsubResource(copyCmd, self.image,
                vk.VK_IMAGE_LAYOUT_UNDEFINED,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresourceRange
            )
            vk.vkCmdCopyBufferToImage(copyCmd, stagingBuffer, self.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                len(bufferCopyRegions), bufferCopyRegions)
            self.imageLayout = imageLayout
            vks.vulkantools.setImageLayoutsubResource(copyCmd, self.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                imageLayout,
                subresourceRange
            )
            self.device.flushCommandBuffer(copyCmd, copyQueue)
            vk.vkFreeMemory(self.device.logicalDevice, stagingMemory, None)
            vk.vkDestroyBuffer(self.device.logicalDevice, stagingBuffer, None)
        else:
            raise(NotImplementedError, 'Non staging buffers are not implemented yet')
        # Create a defaultsampler
        samplerCreateInfo = vk.VkSamplerCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter = vk.VK_FILTER_LINEAR,
            minFilter = vk.VK_FILTER_LINEAR,
            mipmapMode = vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
            addressModeU = vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeV = vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeW = vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            mipLodBias = 0.0,
            compareOp = vk.VK_COMPARE_OP_NEVER,
            minLod = 0.0,
            maxLod = self.mipLevels if useStaging else 0.0,
            anisotropyEnable = self.device.enabledFeatures.samplerAnisotropy,
            maxAnisotropy = self.device.properties.limits.maxSamplerAnisotropy if self.device.enabledFeatures.samplerAnisotropy else 1.0,
            borderColor = vk.VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
        )
        self.sampler = vk.vkCreateSampler(self.device.logicalDevice, samplerCreateInfo, None)
        # Create image view
        # Textures are not directly accessed by the shaders and
        # are abstracted by image views containing additional
        # information and sub resource ranges
        subresourceRange = vk.VkImageSubresourceRange(
            aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel = 0,
            levelCount = self.mipLevels if useStaging else 1,
            baseArrayLayer = 0,
            layerCount = 1
        )
        components = vk.VkComponentMapping(
            r = vk.VK_COMPONENT_SWIZZLE_R,
            g = vk.VK_COMPONENT_SWIZZLE_G,
            b = vk.VK_COMPONENT_SWIZZLE_B,
            a = vk.VK_COMPONENT_SWIZZLE_A
        )
        viewCreateInfo = vk.VkImageViewCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            viewType = vk.VK_IMAGE_VIEW_TYPE_2D,
            image = self.image,
            format = format,
            subresourceRange = subresourceRange,
            components = components
        )
        self.view = vk.vkCreateImageView(self.device.logicalDevice, viewCreateInfo, None)
        # Update descriptor image info member that can be used for setting up descriptor sets
        self.updateDescriptor()
