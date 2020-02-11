# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk

from vks.vulkanglobals import *

import vks.vulkantools

class SwapChain:
    """
Class wrapping access to the swap chain
  A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
    """
    def __init__(self):
        self.instance = None
        self.device = None
        self.physicalDevice = None
        self.surface = None
        self.pGetPhysicalDeviceSurfaceSupportKHR = vk.VK_NULL_HANDLE
        self.fpGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.VK_NULL_HANDLE
        self.fpGetPhysicalDeviceSurfaceFormatsKHR = vk.VK_NULL_HANDLE
        self.fpGetPhysicalDeviceSurfacePresentModesKHR = vk.VK_NULL_HANDLE
        self.fpCreateSwapchainKHR = vk.VK_NULL_HANDLE
        self.fpDestroySwapchainKHR = vk.VK_NULL_HANDLE
        self.fpGetSwapchainImagesKHR = vk.VK_NULL_HANDLE
        self.fpAcquireNextImageKHR = vk.VK_NULL_HANDLE
        self.fpQueuePresentKHR = vk.VK_NULL_HANDLE
        self.colorFormat = None
        self.colorSpace = None
        # @brief Handle to the current swap chain, required for recreation
        self.swapChain = vk.VK_NULL_HANDLE
        self.imageCount = 0
        self.images = []
        self.imageViews = [] # in place of buffers in SaschaWillems code
        # @brief Queue family index of the detected graphics and presenting device queue
        self.queueNodeIndex = None # UINT32MAX

    def connect(self, instance, physicalDevice, device):
        self.instance = instance
        self.device = device
        self.physicalDevice = physicalDevice
        self.fpGetPhysicalDeviceSurfaceSupportKHR = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR")
        self.fpGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
        self.fpGetPhysicalDeviceSurfaceFormatsKHR = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR")
        self.fpGetPhysicalDeviceSurfacePresentModesKHR = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR")
        self.fpCreateSwapchainKHR = vk.vkGetDeviceProcAddr(self.device, "vkCreateSwapchainKHR")
        self.fpDestroySwapchainKHR = vk.vkGetDeviceProcAddr(self.device, "vkDestroySwapchainKHR")
        self.fpGetSwapchainImagesKHR = vk.vkGetDeviceProcAddr(self.device, "vkGetSwapchainImagesKHR")
        self.fpAcquireNextImageKHR = vk.vkGetDeviceProcAddr(self.device, "vkAcquireNextImageKHR")
        self.fpQueuePresentKHR = vk.vkGetDeviceProcAddr(self.device, "vkQueuePresentKHR")

    def initSurface(self, **kwargs):
        """
Create the os-specific surface
see also https://github.com/gabdube/python-vulkan-triangle for surface management with xcb/wayland/xlib
        """
        if VK_USE_PLATFORM_XCB_KHR:
            connection = kwargs['connection']
            window = kwargs['window']
            pCreateXcbSurfaceKHR = vk.vkGetInstanceProcAddr(self.instance, "vkCreateXcbSurfaceKHR")
            surfaceCreateInfo = vk.VkXcbSurfaceCreateInfoKHR(
                sType = vk.VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                #connection = connection,
                connection = vk.ffi.cast('void*', connection._conn),
                window = window)
            self.surface = pCreateXcbSurfaceKHR(instance=self.instance, pCreateInfo=surfaceCreateInfo, pAllocator=None)
        if VK_USE_PLATFORM_WIN32_KHR:
            platformHandle = kwargs['platformHandle']
            platformWindow = kwargs['platformWindow']
            pCreateWin32SurfaceKHR = vk.vkGetInstanceProcAddr(self.instance, "vkCreateWin32SurfaceKHR")
            surfaceCreateInfo = vk.VkWin32SurfaceCreateInfoKHR(
                sType = vk.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                hinstance = platformHandle,
                hwnd = platformWindow)
            self.surface = pCreateWin32SurfaceKHR(instance=self.instance, pCreateInfo=surfaceCreateInfo, pAllocator = None)
        
        if self.surface is None:
            vks.vulkantools.exitFatal("Could not create surface", -1)
            
        # Get available queue family properties
        queueProps = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physicalDevice)
        # Iterate over each queue to learn whether it supports presenting:
        # Find a queue with present support
        # Will be used to present the swap chain images to the windowing system

        supportsPresent = []
        for i in range(len(queueProps)):
            supportsPresent.append(self.fpGetPhysicalDeviceSurfaceSupportKHR(
                physicalDevice = self.physicalDevice,
                queueFamilyIndex = i,
                surface = self.surface))
        # Search for a graphics and a present queue in the array of queue
        # families, try to find one that supports both
        graphicsQueueNodeIndex = None
        presentQueueNodeIndex = None

        for i in range(len(queueProps)):
#            print('properties for queue family ' + str(i) + ': ', end='')
#            if (queueProps[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT) != 0:
#                print('GRAPHICS', end='')
#            if (queueProps[i].queueFlags & vk.VK_QUEUE_COMPUTE_BIT) != 0:
#                print(', COMPUTE', end='')
#            if (queueProps[i].queueFlags & vk.VK_QUEUE_TRANSFER_BIT) != 0:
#                print(', TRANSFER', end='')
#            if (queueProps[i].queueFlags & vk.VK_QUEUE_SPARSE_BINDING_BIT) != 0:
#                print(', SPARSE_BINDING', end='')
#            if supportsPresent[i] == vk.VK_TRUE:
#                 print(', SUPPORTS PRESENTATION', end='')
#            print(', supportsPresent='+str(supportsPresent[i]))
            if (queueProps[i].queueFlags & vk.VK_QUEUE_GRAPHICS_BIT) != 0:
                if graphicsQueueNodeIndex is None:
                    graphicsQueueNodeIndex = i
                if supportsPresent[i] == vk.VK_TRUE:
                    graphicsQueueNodeIndex = i
                    presentQueueNodeIndex = i
                    break
        if presentQueueNodeIndex is None:
            # If there's no queue that supports both present and graphics
            # try to find a separate present queue
            for i in range(len(queueProps)):
                if supportsPresent[i] == vk.VK_TRUE:
                    presentQueueNodeIndex = i
                    break
#        if VK_USE_PLATFORM_WIN32_KHR and presentQueueNodeIndex is None:   # is this a bug in Win32? look at the python code in sdl2_exmaple
#            presentQueueNodeIndex = graphicsQueueNodeIndex
        # Exit if either a graphics or a presenting queue hasn't been found
        if graphicsQueueNodeIndex is None or presentQueueNodeIndex is None:
            vks.vulkantools.exitFatal("Could not find a graphics and/or presenting queue!", -1)
        # todo : Add support for separate graphics and presenting queue
        if graphicsQueueNodeIndex != presentQueueNodeIndex:
            vks.vulkantools.exitFatal("Separate graphics and presenting queues are not supported yet!", -1)

        self.queueNodeIndex = graphicsQueueNodeIndex

        # Get list of supported surface formats
        surfaceFormats = self.fpGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice=self.physicalDevice, surface=self.surface)
        if (len(surfaceFormats) == 1) and (surfaceFormats[0].format == vk.VK_FORMAT_UNDEFINED):
            self.colorFormat = vk.VK_FORMAT_B8G8R8A8_UNORM
            self.colorSpace = surfaceFormats[0].colorSpace
        else:
            # iterate over the list of available surface format and
            # check for the presence of VK_FORMAT_B8G8R8A8_UNORM
            found_B8G8R8A8_UNORM = False
            for surfaceFormat in surfaceFormats:
                if surfaceFormat.format == vk.VK_FORMAT_B8G8R8A8_UNORM:
                    self.colorFormat = surfaceFormat.format
                    self.colorSpace = surfaceFormat.colorSpace
                    found_B8G8R8A8_UNORM = True
                    break
            # in case VK_FORMAT_B8G8R8A8_UNORM is not available
            # select the first available color format
            if  not found_B8G8R8A8_UNORM:
                self.colorFormat = surfaceFormats[0].format
                self.colorSpace = surfaceFormats[0].colorSpace
    def create(self, width, height, vsync = False):
        """
Create the swapchain and get it's images with given width and height

* @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
* @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
* @param vsync (Optional) Can be used to force vsync'd rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation mode)
* @ returns (width, height) tuple of adjusted width/height
        """
        oldSwapchain = self.swapChain
        # Get physical device surface properties and formats
        surfCaps = self.fpGetPhysicalDeviceSurfaceCapabilitiesKHR(self.physicalDevice, self.surface)
        # Get available present modes
        presentModes = self.fpGetPhysicalDeviceSurfacePresentModesKHR(self.physicalDevice, self.surface)
        assert(len(presentModes) > 0)

        # If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
        swapchainExtent = (width, height)
        # If the surface size is defined, the swap chain size must match
        if surfCaps.currentExtent.width != -1:
            swapchainExtent = (surfCaps.currentExtent.width, surfCaps.currentExtent.height)

        # Select a present mode for the swapchain
        # The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
        # This mode waits for the vertical blank ("v-sync")
        swapchainPresentMode = vk.VK_PRESENT_MODE_FIFO_KHR
        # If v-sync is not requested, try to find a mailbox mode
        # It's the lowest latency non-tearing present mode available
        if (not vsync):
            for i in range(len(presentModes)):
                if presentModes[i] == vk.VK_PRESENT_MODE_MAILBOX_KHR:
                    swapchainPresentMode = vk.VK_PRESENT_MODE_MAILBOX_KHR
                    break
                if swapchainPresentMode != vk.VK_PRESENT_MODE_MAILBOX_KHR and presentModes[i] == vk.VK_PRESENT_MODE_IMMEDIATE_KHR:
                    swapchainPresentMode = vk.VK_PRESENT_MODE_IMMEDIATE_KHR
        # Determine the number of images
        desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1
        if (surfCaps.maxImageCount > 0) and (desiredNumberOfSwapchainImages > surfCaps.maxImageCount):
            desiredNumberOfSwapchainImages = surfCaps.maxImageCount

        # Find the transformation of the surface
        if surfCaps.supportedTransforms & vk.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
            preTransform = vk.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        else:
            preTransform = surfCaps.cuurentTransform

        # Find a supported composite alpha format (not all devices support alpha opaque)
        compositeAlpha = vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        # Simply select the first composite alpha format available
        compositeAlphaFlags = [
            vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            vk.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            vk.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            vk.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
            ]
        for compositeAlphaFlag in compositeAlphaFlags:
            if surfCaps.supportedCompositeAlpha & compositeAlphaFlag:
                compositeAlpha = compositeAlphaFlag
                break

        imageUsage = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        # Enable transfer source on swap chain images if supported
        if surfCaps.supportedUsageFlags & vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
            imageUsage |= vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        # Enable transfer destination on swap chain images if supported
        if surfCaps.supportedUsageFlags & vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT:
            imageUsage |= vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT

        swapchainCI = vk.VkSwapchainCreateInfoKHR(
            sType = vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext = None,
            surface = self.surface,
            minImageCount = desiredNumberOfSwapchainImages,
            imageFormat = self.colorFormat,
            imageColorSpace = self.colorSpace,
            imageExtent = swapchainExtent,
            imageUsage = imageUsage,
            preTransform = preTransform,
            imageArrayLayers = 1,
            imageSharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount = 0,
            pQueueFamilyIndices = None,
            presentMode = swapchainPresentMode,
            oldSwapchain = oldSwapchain,
            # Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
            clipped = vk.VK_TRUE,
            compositeAlpha = compositeAlpha
        )
        self.swapChain = self.fpCreateSwapchainKHR(self.device, swapchainCI, None)

        # If an existing swap chain is re-created, destroy the old swap chain
        # This also cleans up all the presentable images
        if oldSwapchain != vk.VK_NULL_HANDLE:
            for imageView in self.imageViews:
                vk.vkDestroyImageView(self.device, imageView, None)
            self.fpDestroySwapchainKHR(self.device, oldSwapchain, None)

        # Get the swap chain images
        self.images = self.fpGetSwapchainImagesKHR(self.device, self.swapChain)
        self.imageCount = len(self.images)

        # Get the swap chain buffers containing the image and imageview
        for image in self.images:
            subresourceRange = vk.VkImageSubresourceRange(
                aspectMask = vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1)
            # was VK_COMPONENT_SWIZZLE_R/G/B/A in SaschaWillems code
            components = vk.VkComponentMapping(
                r = vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g = vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b = vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a = vk.VK_COMPONENT_SWIZZLE_IDENTITY)

            colorAttachmentView = vk.VkImageViewCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext = None,
                image = image,
                flags = 0,
                viewType = vk.VK_IMAGE_VIEW_TYPE_2D,
                format = self.colorFormat,
                components=components,
                subresourceRange=subresourceRange)

            self.imageViews.append(vk.vkCreateImageView(self.device, colorAttachmentView, None))
        # return adjusted width/height
        return swapchainExtent

    def acquireNextImage(self, presentCompleteSemaphore, imageIndex):
        """
Acquires the next image in the swap chain
@param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
@param imageIndex Pointer to the image index that will be increased if the next image could be acquired
@note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
@return VkResult of the image acquisition
        """
        # By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
        # With that we don't have to handle VK_NOT_READY
        return self.fpAcquireNextImageKHR(self.device, self.swapChain, vk.UINT64_MAX, presentCompleteSemaphore, None)

    def queuePresent(self, queue, imageIndex, waitSemaphore = vk.VK_NULL_HANDLE):
        if waitSemaphore != vk.VK_NULL_HANDLE:
            presentInfo = vk.VkPresentInfoKHR(
                sType = vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext = None,
                swapchainCount = 1,
                pSwapchains = [ self.swapChain ],
                pImageIndices = [ imageIndex ],
                pWaitSemaphores = [ waitSemaphore ],
                waitSemaphoreCount = 1
            )
        else:
            presentInfo = vk.VkPresentInfoKHR(
                sType = vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext = None,
                swapchainCount = 1,
                pSwapchains = [ self.swapChain ],
                pImageIndices = [ imageIndex ]
            )
        return self.fpQueuePresentKHR(queue, presentInfo)

    def cleanup(self):
        """
Destroy and free Vulkan resources used for the swapchain
        """
        if self.swapChain != vk.VK_NULL_HANDLE:
            for imageView in self.imageViews:
                vk.vkDestroyImageView(self.device, imageView, None)
        if self.surface != vk.VK_NULL_HANDLE:
            self.fpDestroySwapchainKHR(self.device, self.swapChain, None)
            fpDestroySurfaceKHR = vk.vkGetInstanceProcAddr(self.instance, "vkDestroySurfaceKHR")
            fpDestroySurfaceKHR(self.instance, self.surface, None)
        self.swapChain = vk.VK_NULL_HANDLE
        self.surface = vk.VK_NULL_HANDLE
