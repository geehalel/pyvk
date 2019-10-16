# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import sys
import argparse
import logging
import os
import platform

import time # for perf_counter()


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger('VulkanExampleBase')

### from vulkanexamplebase.cpp/.h

import vulkan as vk
import glm
import imgui

import vks.vulkanbenchmark
import vks.vulkancamera
import vks.vulkandebug
import vks.vulkandebugmarker
import vks.vulkanswapchain
import vks.vulkantools
import vks.vulkandevice
import vks.vulkanuioverlay

from vks.vulkanglobals import  *
from vks.vulkanglobals import  _WIN32
from vks.vulkanglobals import  _DIRECT2DISPLAY

if VK_USE_PLATFORM_XCB_KHR:
    import xcffib
    import xcffib.xproto
    #import vks.xcb_keyboard
    #import vks.xkeysyms
    #import vks.xkkeys
    import vks.vulkankeycodes

class VulkanExampleBase:

    def getAssetPath(self):
        return './data/' # while not ../data ?

    def createInstance(self, enableValidation=False):
        self.settings['validation'] = enableValidation
        # Validation can also be forced via a define
        try:
            if _VALIDATION is not None:
                self.settings['validation'] = True
        except:
            pass
        appInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.name,
            pEngineName=self.name,
            apiVersion=self.apiVersion)

        # Info added in realtix/vulkan
        #extensions = vk.vkEnumerateInstanceExtensionProperties(None)
        #extensions = [e.extensionName for e in extensions]
        #print("Available extensions: %s\n" % extensions)
        #layers = vk.vkEnumerateInstanceLayerProperties()
        #layers = [l.layerName for l in layers]
        #print("Available layers: %s\n" % layers)

        instanceExtensions = [ vk.VK_KHR_SURFACE_EXTENSION_NAME ]
        if _WIN32:
            instanceExtensions.append(vk.VK_KHR_WIN32_SURFACE_EXTENSION_NAME)
        elif VK_USE_PLATFORM_ANDROID_KHR:
            instanceExtensions.append(vk.VK_KHR_ANDROID_SURFACE_EXTENSION_NAME)
        elif _DIRECT2DISPLAY:
            instanceExtensions.append(vk.VK_KHR_DISPLAY_EXTENSION_NAME)
        elif VK_USE_PLATFORM_WAYLAND_KHR:
            instanceExtensions.append(vk.VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME)
        elif VK_USE_PLATFORM_XCB_KHR:
            instanceExtensions.append(vk.VK_KHR_XCB_SURFACE_EXTENSION_NAME);
        #elif VK_USE_PLATFORM_IOS_MVK:
        #    instanceExtensions.append(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
        #elif VK_USE_PLATFORM_MACOS_MVK:
        #    instanceExtensions.append(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
        if len(self.enabledInstanceExtensions) > 0:
            instanceExtensions.extend(self.enabledInstanceExtensions)
        if len(instanceExtensions) > 0:
            if self.settings['validation']:
                instanceExtensions.append(vk.VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        layers = []
        if self.settings['validation']:
            layers = vks.vulkandebug.validationLayerNames
        print("Using extensions: %s" % instanceExtensions)
        print("Using layers: %s\n" % layers)
        instanceCreateInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=0,
            pApplicationInfo=appInfo,
            enabledExtensionCount=len(instanceExtensions),
            ppEnabledExtensionNames=instanceExtensions,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers)

        return vk.vkCreateInstance(instanceCreateInfo, None)

    def getWindowTitle(self):
        windowTitle = self.title + " - " + self.deviceProperties.deviceName
        if not self.settings['overlay']:
            windowTitle += " - " + str(self.frameCounter) + " fps"
        return windowTitle

    def initSwapchain(self):
        if _WIN32:
            self.swapChain.initSurface(self.windowInstance, self.window)
        elif VK_USE_PLATFORM_ANDROID_KHR:
            self.swapChain.initSurface(self.androidApp.window)
        #elif VK_USE_PLATFORM_IOS_MVK or VK_USE_PLATFORM_MACOS_MVK:
        #    self.swapChain.initSurface(self.view)
        elif _DIRECT2DISPLAY:
            self.swapChain.initSurface(self.width, self.height)
        elif VK_USE_PLATFORM_WAYLAND_KHR:
            self.swapChain.initSurface(self.display, self.surface)
        elif VK_USE_PLATFORM_XCB_KHR:
            self.swapChain.initSurface(connection=self.connection, window=self.window)

    def createCommandPool(self):
        cmdPoolInfo = vk.VkCommandPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex = self.swapChain.queueNodeIndex,
            flags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
        self.cmdPool = vk.vkCreateCommandPool(self.device, cmdPoolInfo, None)

    def setupSwapChain(self):
        self.width, self.height = self.swapChain.create(self.width, self.height, self.settings['vsync'])

    def createCommandBuffers(self):
        """
Create one command buffer for each swap chain image and reuse for rendering
        """
        cmdBufAllocateInfo = vk.VkCommandBufferAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool = self.cmdPool,
            level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount = self.swapChain.imageCount)

        self.drawCmdBuffers = vk.vkAllocateCommandBuffers(self.device, cmdBufAllocateInfo)

    def destroyCommandBuffers(self):
        vk.vkFreeCommandBuffers(self.device, self.cmdPool, len(self.drawCmdBuffers), self.drawCmdBuffers)

    def createSynchronizationPrimitives(self):
        # Wait fences to sync command buffer access
        fenceCreateInfo = vk.VkFenceCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags = vk.VK_FENCE_CREATE_SIGNALED_BIT
        )
        self.waitFences = []
        for _ in range(len(self.drawCmdBuffers)):
            self.waitFences.append(vk.vkCreateFence(self.device, fenceCreateInfo, None))
    def setupDepthStencil(self):
        imageCI = vk.VkImageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType = vk.VK_IMAGE_TYPE_2D,
            format = self.depthFormat,
            extent = (self.width, self.height, 1),
            mipLevels = 1,
            arrayLayers = 1,
            samples = vk.VK_SAMPLE_COUNT_1_BIT,
            tiling = vk.VK_IMAGE_TILING_OPTIMAL,
            usage = vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        )
        self.depthStencil['image'] = vk.vkCreateImage(self.device, imageCI, None)
        memReqs = vk.vkGetImageMemoryRequirements(self.device, self.depthStencil['image'])
        memAlloc = vk.VkMemoryAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize = memReqs.size,
            memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        )
        self.depthStencil['mem'] = vk.vkAllocateMemory(self.device, memAlloc, None)
        vk.vkBindImageMemory(self.device, self.depthStencil['image'], self.depthStencil['mem'], 0)
        aspectMask = vk.VK_IMAGE_ASPECT_DEPTH_BIT
        if self.depthFormat >= vk.VK_FORMAT_D16_UNORM_S8_UINT:
            aspectMask |= vk.VK_IMAGE_ASPECT_STENCIL_BIT
        subresourceRange = vk.VkImageSubresourceRange(
            baseMipLevel = 0,
            levelCount = 1,
            baseArrayLayer = 0,
            layerCount = 1,
            aspectMask = aspectMask
        )
        imageViewCI = vk.VkImageViewCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            viewType = vk.VK_IMAGE_VIEW_TYPE_2D,
            image = self.depthStencil['image'],
            format = self.depthFormat,
            subresourceRange = subresourceRange
        )
        self.depthStencil['view'] = vk.vkCreateImageView(self.device, imageViewCI, None)

    def setupRenderPass(self):
        attachments =[]
        attachment = vk.VkAttachmentDescription(
            format = self.swapChain.colorFormat,
            samples = vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout = vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        attachments.append(attachment)
        attachment = vk.VkAttachmentDescription(
            format = self.depthFormat,
            samples = vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp = vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp = vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            stencilStoreOp = vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout = vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        attachments.append(attachment)
        colorReference = vk.VkAttachmentReference(
            attachment = 0,
            layout = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        depthReference = vk.VkAttachmentReference(
            attachment = 1,
            layout = vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            )
        subpassDescription = vk.VkSubpassDescription(
            pipelineBindPoint = vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount = 1,
            pColorAttachments = colorReference,
            pDepthStencilAttachment = depthReference,
            inputAttachmentCount = 0,
            pInputAttachments = None,
            preserveAttachmentCount = 0,
            pPreserveAttachments = None,
            pResolveAttachments = None
        )
        subPasses = []
        subPasses.append(subpassDescription)
        dependencies = []
        dependency = vk.VkSubpassDependency(
            srcSubpass = vk.VK_SUBPASS_EXTERNAL,
            dstSubpass = 0,
            srcStageMask = vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            dstStageMask = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask = vk.VK_ACCESS_MEMORY_READ_BIT,
            dstAccessMask = vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dependencyFlags = vk.VK_DEPENDENCY_BY_REGION_BIT
        )
        dependencies.append(dependency)
        dependency = vk.VkSubpassDependency(
            srcSubpass = 0,
            dstSubpass = vk.VK_SUBPASS_EXTERNAL,
            srcStageMask = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask = vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            srcAccessMask = vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask = vk.VK_ACCESS_MEMORY_READ_BIT,
            dependencyFlags = vk.VK_DEPENDENCY_BY_REGION_BIT
        )
        dependencies.append(dependency)
        renderPassInfo = vk.VkRenderPassCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount = len(attachments),
            pAttachments = attachments,
            subpassCount = 1,
            pSubpasses = subPasses,
            dependencyCount = len(dependencies),
            pDependencies = dependencies
        )
        self.renderPass = vk.vkCreateRenderPass(self.device, renderPassInfo, None)

    def createPipelineCache(self):
        pipelineCacheCreateInfo = vk.VkPipelineCacheCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
        )
        self.pipelineCache = vk.vkCreatePipelineCache(self.device, pipelineCacheCreateInfo, None)

    def setupFrameBuffer(self):
        # Depth/Stencil attachment is the same for all frame buffers
        # attachments = [0, self.depthStencil['view']]
        # frameBufferCreateInfo = vk.VkFramebufferCreateInfo(
        #     sType = vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        #     pNext = None,
        #     renderPass = self.renderPass,
        #     attachmentCount = len(attachments),
        #     pAttachments = attachments,
        #     width = self.width,
        #     height = self.height,
        #     layers = 1
        # )
        # Create frame buffers for every swap chain image
        self.frameBuffers = []
        for imageView in self.swapChain.imageViews:
            attachments = [imageView, self.depthStencil['view']]
            frameBufferCreateInfo = vk.VkFramebufferCreateInfo(
                sType = vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext = None,
                renderPass = self.renderPass,
                attachmentCount = len(attachments),
                pAttachments = attachments,
                width = self.width,
                height = self.height,
                layers = 1
            )
            self.frameBuffers.append(vk.vkCreateFramebuffer(self.device, frameBufferCreateInfo, None))

    def loadShader(self, filename, stage):
        module = vks.vulkantools.loadShader(filename, self.device)
        shaderStage = vk.VkPipelineShaderStageCreateInfo (
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage = stage,
            module = module,
            pName = "main"
        )
        self.shaderModules.append(module)
        return shaderStage

    def prepare(self):
        if self.vulkanDevice.enableDebugMarkers:
            vks.vulkandebugmarker.setup(self.device)
        self.initSwapchain()
        self.createCommandPool()
        self.setupSwapChain()
        self.createCommandBuffers()
        self.createSynchronizationPrimitives()
        self.setupDepthStencil()
        self.setupRenderPass()
        self.createPipelineCache()
        self.setupFrameBuffer()
        self.settings['overlay'] = self.settings['overlay'] and not(self.benchmark.active)
        if self.settings['overlay']:
            self.UIOverlay.device = self.vulkanDevice
            self.UIOverlay.queue = self.queue
            self.UIOverlay.shaders.append(self.loadShader(self.getAssetPath() +"shaders/base/uioverlay.vert.spv", vk.VK_SHADER_STAGE_VERTEX_BIT))
            self.UIOverlay.shaders.append(self.loadShader(self.getAssetPath() +"shaders/base/uioverlay.frag.spv", vk.VK_SHADER_STAGE_FRAGMENT_BIT))
            self.UIOverlay.prepareResources()
            self.UIOverlay.preparePipeline(self.pipelineCache, self.renderPass)

    def __init__(self, enableValidation=False):
        import sys # why should I put it there ?
        self._viewUpdated = False
        self._destWidth = 0
        self._destHeight = 0
        self._resizing = False
        # protected
        self.frameCounter = 0
        self.lastFPS = 0
        self.lastTimestamp =0.0
        self.instance = None
        self.physicalDevice = None
        self.deviceProperties = None
        self.deviceFeatures = None
        self.deviceMemoryProperties = None
        self.enabledFeatures = vk.VkPhysicalDeviceFeatures()
        self.enabledDeviceExtensions = []
        self.enabledInstanceExtensions = []
        self.device = None
        self.queue= None
        self.depthFormat = None
        self.cmdPool = None
        self.submitPipelineStages = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        self.submitInfo = None
        self.drawCmdBuffers = []
        self.renderPass = None
        self.frameBuffers = []
        self.currentBuffer = 0
        self.descriptorPool = vk.VK_NULL_HANDLE
        self.shaderModules = []
        self.pipelineCache = None
        self.swapChain = vks.vulkanswapchain.SwapChain()
        self.semaphores = {'presentComplete': None, 'renderComplete': None}
        self.waitFences = []
        #public
        self.prepared = False
        self.width = 1280
        self.height = 720
        self.UIOverlay = None
        self.frameTimer = 1.0
        self.benchmark = vks.vulkanbenchmark.Benchmark()
        self.vulkanDevice = None
        self.settings = { 'validation': False, 'fullscreen': False, 'vsync': False, 'overlay': False }
        self.defaultClearColor = vk.VkClearColorValue(float32=[0.025, 0.025, 0.025, 1.0])
        self.zoom = 0
        self.timer = 0.0
        self.timerSpeed = 0.25
        self.paused = False
        self.rotationSpeed = 1.0
        self.zoomSpeed = 1.0
        self.camera = vks.vulkancamera.Camera()
        self.rotation = glm.vec3()
        self.cameraPos = glm.vec3()
        self.mousePos = glm.vec2()
        self.title = "Vulkan Example"
        self.name = "vulkanexample"
        self.apiVersion = vk.VK_API_VERSION_1_0
        self.depthStencil = {'image': None, 'mem': None, 'view': None}
        self.gamePadState = {'axisLeft':glm.vec2(0.0), 'axisRight':glm.vec2(0.0)}
        self.mouseButtons = {'left': False, 'right': False, 'middle': False}
        self.UIOverlay = vks.vulkanuioverlay.UIOverlay()
        if _WIN32 or VK_USE_PLATFORM_ANDROID_KHR or VK_USE_PLATFORM_WAYLAND_KHR or _DIRECT2DISPLAY:
            import sys
            print('Platform not supported')
            sys.exit(1)
        if VK_USE_PLATFORM_XCB_KHR:
            self.quit = False
            self.connection = None
            self.screen = None
            self.window = None
            self.atom_wm_delete_window = None
        # Ctor
        if not VK_USE_PLATFORM_ANDROID_KHR:
            # Check for a valid asset path
            if not os.path.exists(self.getAssetPath()):
                if not _WIN32:
                    print("Error: Could not find asset path in ", self.getAssetPath())#, file=sys.stderr)
                sys.exit(-1)
        self.settings['validation'] = enableValidation
        # Parse command line arguments
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('-validation', action='store_true')
        parser.add_argument('-vsync', action='store_true')
        parser.add_argument('-f', '--fullscreen', action='store_true')
        parser.add_argument('-w', '--width', type=int)
        parser.add_argument('-h', '--height', type=int)
        parser.add_argument('-b', '--benchmark', action='store_true')
        parser.add_argument('-bw', '--benchwarmup', type=int)
        parser.add_argument('-br', '--benchruntime', type=int)
        parser.add_argument('-bf', '--benchfilename', type=str)
        parser.add_argument('-bt', '--benchframetimes', action='store_true')
        parser.add_argument('-g', '--gpu', type = int)
        parser.add_argument('-listgpus', action='store_true')
        pargs=parser.parse_args()
        if pargs.validation: self.settings['validation'] = True
        if pargs.vsync: self.settings['vsync'] = True
        if pargs.fullscreen: self.settings['fullscreen'] = True
        if pargs.width: self.width = pargs.width
        if pargs.height: self.height = pargs.height
        if pargs.benchmark: self.benchmark.active = True
        if pargs.benchwarmup: self.benchmark.warmup = pargs.benchwarmup
        if pargs.benchruntime: self.benchmark.runtime = pargs.benchruntime
        if pargs.benchfilename: self.benchmark.filename = pargs.benchfilename
        if pargs.benchframetimes: self.benchmark.outputFrameTimes = True
        self.args = pargs

        if VK_USE_PLATFORM_XCB_KHR:
            self.initxcbConnection()

    # use __del__ as Dtor
    def __del__(self):
        if self.swapChain:
            self.swapChain.cleanup()
        if self.descriptorPool != vk.VK_NULL_HANDLE:
            vk.vkDestroyDescriptorPool(self.device, self.descriptorPool, None)
        self.destroyCommandBuffers()
        vk.vkDestroyRenderPass(self.device, self.renderPass, None)
        for i in range(len(self.frameBuffers)):
            vk.vkDestroyFramebuffer(self.device, self.frameBuffers[i], None)
        for shaderModule in self.shaderModules:
            vk.vkDestroyShaderModule(self.device, shaderModule)
        vk.vkDestroyImageView(self.device, self.depthStencil['view'], None)
        vk.vkDestroyImage(self.device, self.depthStencil['image'], None)
        vk.vkFreeMemory(self.device, self.depthStencil['mem'], None)

        vk.vkDestroyPipelineCache(self.device, self.pipelineCache, None)

        vk.vkDestroyCommandPool(self.device, self.cmdPool, None)

        vk.vkDestroySemaphore(self.device, self.semaphores['presentComplete'], None)
        vk.vkDestroySemaphore(self.device, self.semaphores['renderComplete'], None)
        for fence in self.waitFences:
            vk.vkDestroyFence(self.device, fence, None)

        if self.settings['overlay']:
            UIOverlay.freeResources()

        self.vulkanDevice.cleanup()
        self.vulkanDevice = None

        if self.settings['validation']:
            vks.vulkandebug.freeDebugCallBack(self.instance)

        vk.vkDestroyInstance(self.instance, None)

        if VK_USE_PLATFORM_XCB_KHR:
            self.connection.core.DestroyWindow(self.window)
            self.connection.disconnect()

    def initVulkan(self):
        try:
            self.instance = self.createInstance(self.settings['validation'])
        except vk.VkError as e:
            print(e)
            print("Unable to create vulkan instance")
            return False

        if self.settings['validation']:
            debugReportFlags = vk.VK_DEBUG_REPORT_ERROR_BIT_EXT | vk.VK_DEBUG_REPORT_WARNING_BIT_EXT
            vks.vulkandebug.setupDebugging(self.instance, debugReportFlags, vk.VK_NULL_HANDLE)

        try:
            physicalDevices = vk.vkEnumeratePhysicalDevices(self.instance)
        except VkError:
            return False

        if self.args.listgpus:
            if len(physicalDevices) == 0:
                print('No Vulkan devices found')
            else:
                print('Available Vulkan devices:')
                i = 0
                for d in physicalDevices:
                    deviceProperties = vk.vkGetPhysicalDeviceProperties(d)
                    print('Device [',i,'] : ', deviceProperties.deviceName, sep='')
                    print(' Type: ', vks.vulkantools.physicalDeviceTypeString(deviceProperties.deviceType), sep='')
                    print(' API: ', deviceProperties.apiVersion >> 22, '.', ((deviceProperties.apiVersion >> 12) & 0x3ff), '.',  ((deviceProperties.apiVersion) & 0xfff), sep='')

        selectedDevice = 0
        if self.args.gpu:
            selectedDevice = self.args.gpu
        if selectedDevice >= len(physicalDevices):
            selectedDevice = 0

        self.physicalDevice = physicalDevices[selectedDevice]
        try:
            self.deviceProperties = vk.vkGetPhysicalDeviceProperties(self.physicalDevice)
            self.deviceFeatures = vk.vkGetPhysicalDeviceFeatures(self.physicalDevice)
            self.deviceMemoryProperties = vk.vkGetPhysicalDeviceMemoryProperties(self.physicalDevice)
        except vk.VkError as e:
            print(e)
            print("Unable to get physical properties/features/memory properties")
            return False
        #  Derived examples can override this to set actual features (based on above readings) to enable for logical device creation
        self.getEnabledFeatures()
        self.vulkanDevice = vks.vulkandevice.VulkanDevice(self.physicalDevice)
        #try:
        self.vulkanDevice.createLogicalDevice(self.enabledFeatures, self.enabledDeviceExtensions)
        #except Exception as e:
        #    print(e)
        #    print("Unable to create vulkan logical device")
        #    return False
        self.device = self.vulkanDevice.logicalDevice
        self.queue = vk.vkGetDeviceQueue(self.device, self.vulkanDevice.queueFamilyIndices['graphics'], 0)
        self.depthFormat = vks.vulkantools.getSupportedDepthFormat(self.physicalDevice)
        assert(self.depthFormat is not None)

        self.swapChain.connect(self.instance, self.physicalDevice, self.device)

        semaphoreCreateInfo = vk.VkSemaphoreCreateInfo(sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
        # Create a semaphore used to synchronize image presentation
	    # Ensures that the image is displayed before we start submitting new commands to the queu
        self.semaphores['presentComplete'] = vk.vkCreateSemaphore(self.device, semaphoreCreateInfo, None)
        # Create a semaphore used to synchronize command submission
	    # Ensures that the image is not presented until all commands have been sumbitted and executed
        self.semaphores['renderComplete'] = vk.vkCreateSemaphore(self.device, semaphoreCreateInfo, None)

        # Set up submit info structure
	    # Semaphores will stay the same during application lifetime
	    # Command buffer submission info is set by each example
        self.submitInfo = vk.VkSubmitInfo(sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pWaitDstStageMask = self.submitPipelineStages,
            waitSemaphoreCount = 1,
            pWaitSemaphores = [self.semaphores['presentComplete']],
            signalSemaphoreCount = 1,
            pSignalSemaphores = [self.semaphores['renderComplete']])

        return True
    # Keyboard/Mouse stuff
    def handleMouseMove(self, x, y):
        dx = self.mousePos.x - x
        dy = self.mousePos.y - y
        handled = False
        if self.settings['overlay']:
            pass
        handled = self.mouseMoved(x, y)
        if handled:
            self.mousePos = glm.vec2(x, y)
            return
        if self.mouseButtons['left']:
            self.rotation.x += dy * 1.25 * self.rotationSpeed
            self.rotation.y -= dx * 1.25 * self.rotationSpeed
            self.camera.rotate(glm.vec3(dy * self.camera.rotationSpeed, -dx * self.camera.rotationSpeed, 0.0))
            self._viewUpdated = True
        if self.mouseButtons['right']:
            self.zoom += dy * 0.005 *self.zoomSpeed
            self.camera.translate(glm.vec3(0.0, 0.0, dy * 0.005 * self.zoomSpeed))
            self._viewUpdated = True
        if self.mouseButtons['middle']:
            self.cameraPos.x -= dx * 0.01
            self.cameraPos.y -= dy * 0.01
            self.camera.translate(glm.vec3(-dx*0.01, -dy * 0.01, 0.0))
            self._viewUpdated = True
        self.mousePos = glm.vec2(x, y)
    # xcb keyboard
    def handleEvent(self, event):
        #print(event.__class__)
        if event.__class__ == xcffib.xproto.ClientMessageEvent:
            if event.data.data32[0] == self.atom_wm_delete_window.atom:
                self.quit = True
        elif event.__class__ == xcffib.xproto.MotionNotifyEvent:
            self.handleMouseMove(event.event_x, event.event_y)
        elif event.__class__ == xcffib.xproto.ButtonPressEvent:
            if event.detail == 1:
                self.mouseButtons['left'] = True
            elif event.detail == 2:
                self.mouseButtons['middle'] = True
            elif event.detail == 3:
                self.mouseButtons['right'] = True
            elif event.detail == 4:
                self.mouseButtons['wheelup'] = True
            elif event.detail == 5:
                self.mouseButtons['wheeldown'] = True
        elif event.__class__ == xcffib.xproto.ButtonReleaseEvent:
            if event.detail == 1:
                self.mouseButtons['left'] = False
            elif event.detail == 2:
                self.mouseButtons['middle'] = False
            elif event.detail == 3:
                self.mouseButtons['right'] = False
            elif event.detail == 4:
                self.mouseButtons['wheelup'] = False
            elif event.detail == 5:
                self.mouseButtons['wheeldown'] = False
        elif event.__class__ == xcffib.xproto.KeyPressEvent:
            #print(event.detail, self.xcbkeyboard.code_to_syms[event.detail], event.state)
            #evkeysym = self.xcbkeyboard.code_to_syms[event.detail][0]
            #if vks.xkeysyms.keysyms['q'] == evkeysym or vks.xkeysyms.keysyms['Q'] == evkeysym:
            #if evkeysym == vks.xkkeys.XkKeys.XK_q:
            #    self.quit = True
            if event.detail == vks.vulkankeycodes.Keycode.KEY_W:
                self.camera.keys['up'] = True
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_S:
                self.camera.keys['down'] = True
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_A:
                self.camera.keys['left'] = True
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_D:
                self.camera.keys['right'] = True
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_P:
                self.paused = not(self.paused)
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_F1:
                if self.settings['overlay']:
                    self.settings['overlay'] = not(self.settings['overlay'])
        elif event.__class__ == xcffib.xproto.KeyReleaseEvent:
            if event.detail == vks.vulkankeycodes.Keycode.KEY_W:
                self.camera.keys['up'] = False
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_S:
                self.camera.keys['down'] = False
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_A:
                self.camera.keys['left'] = False
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_D:
                self.camera.keys['right'] = False
            elif event.detail == vks.vulkankeycodes.Keycode.KEY_ESCAPE:
                self.quit = True
            self.keyPressed(event.detail, event.state)
        elif event.__class__ == xcffib.xproto.DestroyNotifyEvent:
            self.quit = True

    # xcb stuff
    def initxcbConnection(self):
        if VK_USE_PLATFORM_XCB_KHR:
            import os
            import time
            # for _ in range(100):
            #     try:
            #         #self.connection = xcffib.connect(display=os.environ['DISPLAY'])
            #         self.connection = xcffib.connect()
            #         self.connection.invalid()
            #     except ConnectionException:
            #         print('Retrying Xcb connection')
            #         self.connection.dsiconnect()
            #         time.sleep(0.2)
            self.connection = xcffib.connect()
            if self.connection is None or self.connection.invalid():
                printf("Could not find a compatible Vulkan ICD!")
                sys.exit(1)
            setup = self.connection.get_setup()
            self.screen = setup.roots[self.connection.pref_screen]
            self.xproto = xcffib.xproto.xprotoExtension(self.connection)
    def intern_atom_helper(self, conn, only_if_exists, strname):
        cookie = self.xproto.InternAtom(only_if_exists, len(strname), strname)
        return cookie.reply()


    def setupWindow(self):
        if VK_USE_PLATFORM_XCB_KHR:
            value_mask = 0
            value_list = []
            self.window = self.connection.generate_id()
            value_mask = xcffib.xproto.CW.BackPixel | xcffib.xproto.CW.EventMask
            value_list.append(self.screen.black_pixel)
            value_list.append(xcffib.xproto.EventMask.KeyRelease |
                xcffib.xproto.EventMask.KeyPress |
                xcffib.xproto.EventMask.Exposure |
                xcffib.xproto.EventMask.StructureNotify |
                xcffib.xproto.EventMask.PointerMotion |
                xcffib.xproto.EventMask.ButtonPress |
                xcffib.xproto.EventMask.ButtonRelease)
            if (self.settings['fullscreen']):
                self.width = self.destWidth = self.screen.width_in_pixels
                self.height = self.destHeight = self.screen.height_in_pixels
            self.connection.core.CreateWindow(
                xcffib.XCB_COPY_FROM_PARENT,
                self.window, self.screen.root,
                0, 0, self.width, self.height, 0,
                xcffib.xproto.WindowClass.InputOutput,
                self.screen.root_visual,
                value_mask, value_list)
            # Magic code that will send notification when window is destroyed
            reply = self.intern_atom_helper(self.connection, True, "WM_PROTOCOLS")
            self.atom_wm_delete_window = self.intern_atom_helper(self.connection, False, "WM_DELETE_WINDOW")
            self.connection.core.ChangeProperty(xcffib.xproto.PropMode.Replace,
                self.window, reply.atom, xcffib.xproto.Atom.ATOM, 32, 1,
		        [self.atom_wm_delete_window.atom])
            windowTitle = self.getWindowTitle()
            self.connection.core.ChangeProperty(xcffib.xproto.PropMode.Replace,
                self.window, xcffib.xproto.Atom.WM_NAME, xcffib.xproto.Atom.STRING, 8,
		        len(windowTitle), windowTitle)

            if self.settings['fullscreen']:
                atom_wm_state = self.intern_atom_helper(self.connection, False, "_NET_WM_STATE")
                atom_wm_fullscreen = self.intern_atom_helper(self.connection, False, "_NET_WM_STATE_FULLSCREEN")
                self.connection.core.ChangeProperty(xcffib.xproto.PropMode.Replace,
                    self.window, atom_wm_state.atom, xcffib.xproto.Atom.ATOM, 32, 1,
    		        [atom_wm_fullscreen.atom])
            self.connection.core.MapWindow(self.window)
            self.connection.flush()
            #self.xcbkeyboard = vks.xcb_keyboard.Keyboard(self.connection.setup, self.connection)

    # ImGui stuff
    def updateOverlay(self):
        if not self.settings['overlay']:
            return
        io = imgui.get_io()
        io.display_size = imgui.Vec2(self.width, self.height)
        io.delta_time = self.frameTimer
        io.mouse_pos = imgui.Vec2(self.mousePos.x, self.mousePos.y)
        io.mouse_down[0] = self.mouseButtons['left']
        io.mouse_down[1] = self.mouseButtons['right']

        imgui.new_frame()
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(0, 0, imgui.FIRST_USE_EVER)
        imgui.begin('Vulkan Example', None, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_RESIZE| imgui.WINDOW_NO_MOVE)
        imgui.text_unformatted(self.title)
        imgui.text_unformatted(self.deviceProperties.deviceName)
        if self.lastFPS > 0:
            imgui.text("{:.2f} ms/frame ({:.1f} fps)".format((1000.0 / self.lastFPS), self.lastFPS))
        else:
            imgui.text("--- ms/frame ({:.1f} fps)".format(self.lastFPS))
        imgui.push_item_width(110.0 * self.UIOverlay.scale)
        self.onUpdateUIOverlay(self.UIOverlay)
        imgui.pop_item_width()
        imgui.end()
        imgui.pop_style_var()
        imgui.render()
        if self.UIOverlay.update() or self.UIOverlay.updated:
            self.buildCommandBuffers()
            self.UIOverlay.updated = False

    def drawUI(self, commandBuffer):
        if self.settings['overlay']:
            viewport = vk.VkViewport(
                height = float(self.height),
                width = float(self.width),
                minDepth = 0.0,
                maxDepth = 1.0
            )
            vk.vkCmdSetViewport(commandBuffer, 0, 1, [viewport])
            # Update dynamic scissor state
            offsetscissor = vk.VkOffset2D(x = 0, y = 0)
            extentscissor = vk.VkExtent2D(width = self.width, height = self.height)
            scissor = vk.VkRect2D(offset = offsetscissor, extent = extentscissor)
            vk.vkCmdSetScissor(commandBuffer, 0, 1, [scissor])
            self.UIOverlay.draw(commandBuffer)

    def renderLoop(self):
        if self.benchmark.active:
            # TODO
            return
        self.destWidth = self.width
        self.destHeight = self.height
        self.lastTimestamp = time.monotonic()
        if VK_USE_PLATFORM_XCB_KHR:
            self.connection.flush()
            while not self.quit:
                tStart = time.monotonic()
                if self._viewUpdated:
                    self._viewUpdated = False
                    self.viewChanged()
                event = self.connection.poll_for_event()
                while event is not None:
                    self.handleEvent(event)
                    # TODO free(event)
                    event = self.connection.poll_for_event()
                self.render()
                self.frameCounter += 1
                tEnd = time.monotonic()
                tDiff = (tEnd - tStart) * 1.0E3
                self.frameTimer = tDiff / 1000.0
                # TODO camera stuff
                self.camera.update(self.frameTimer)
                if self.camera.moving():
                    self._viewUpdated = True
                if not self.paused:
                    self.timer += self.timerSpeed * self.frameTimer
                    if self.timer > 1.0:
                        self.timer -= 1.0
                fpsTimer = (tEnd - self.lastTimestamp) * 1.0E3
                if fpsTimer > 1000.0:
                    if not self.settings['overlay']:
                        windowTitle = self.getWindowTitle()
                        self.connection.core.ChangeProperty(xcffib.xproto.PropMode.Replace,
                            self.window, xcffib.xproto.Atom.WM_NAME, xcffib.xproto.Atom.STRING, 8,
            		        len(windowTitle), windowTitle)
                        self.connection.flush()
                    self.lastFPS = self.frameCounter * (1000.0 / fpsTimer)
                    self.frameCounter = 0
                    self.lastTimestamp = tEnd
                self.updateOverlay()
        if self.device != vk.VK_NULL_HANDLE:
            vk.vkDeviceWaitIdle(self.device)

    # Virtual methods
    def getEnabledFeatures(self):
        return
    def render(self):
        pass
    def keyPressed(self, evdetail, evstate):
        pass
    def viewChanged(self):
        pass
    def mouseMoved(self, x, y):
        return False
    def onUpdateUIOverlay(self, overlay):
        pass
    # Main loop
    def main(self):
        r = self.initVulkan()
        print("   *** initVulkan passed: ", r, "***")
        self.setupWindow()
        print("   *** setupWindow passed ***")
        self.prepare()
        print("   *** prepare passed ***")
        self.renderLoop()
        #self.cleanup()
        sys.exit(0)
