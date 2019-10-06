# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk

"""
Setup and functions for the VK_EXT_debug_marker_extension
Extension spec can be found at https://github.com/KhronosGroup/Vulkan-Docs/blob/1.0-VK_EXT_debug_marker/doc/specs/vulkan/appendices/VK_EXT_debug_marker.txt
Note that the extension will only be present if run from an offline debugging application
The actual check for extension presence and enabling it on the device is done in the example base class
See VulkanExampleBase::createInstance and VulkanExampleBase::createDevice (base/vulkanexamplebase.cpp)
"""
active = False

pfnDebugMarkerSetObjectTag = vk.VK_NULL_HANDLE
pfnDebugMarkerSetObjectName = vk.VK_NULL_HANDLE
pfnCmdDebugMarkerBegin = vk.VK_NULL_HANDLE
pfnCmdDebugMarkerEnd = vk.VK_NULL_HANDLE
pfnCmdDebugMarkerInsert = vk.VK_NULL_HANDLE

def setup(device):
    pfnDebugMarkerSetObjectTag = vk.vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectTagEXT")
    pfnDebugMarkerSetObjectName = vk.vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT")
    pfnCmdDebugMarkerBegin = vk.vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT")
    pfnCmdDebugMarkerEnd = vk.vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT")
    pfnCmdDebugMarkerInsert = vk.vkGetDeviceProcAddr(device, "vkCmdDebugMarkerInsertEXT")
    # Set flag if at least one function pointer is present
    active = (pfnDebugMarkerSetObjectName != vk.VK_NULL_HANDLE)
