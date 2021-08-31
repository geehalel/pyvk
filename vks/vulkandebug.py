# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import platform
import sys
# I left the Android case
validationLayerCount = 1
validationLayerNames= [
            #"VK_LAYER_NV_optimus"
            #"VK_LAYER_KHRONOS_validation"
            "VK_LAYER_LUNARG_standard_validation"
        ]
CreateDebugReportCallback = vk.VK_NULL_HANDLE
DestroyDebugReportCallback = vk.VK_NULL_HANDLE;
dbgBreakCallback = vk.VK_NULL_HANDLE

msgCallback = None

def messageCallback(flags, objType, srcObject, location, msgCode, pLayerPrefix, pMsg, pUserData):
    prefix = ""
    if flags & vk.VK_DEBUG_REPORT_ERROR_BIT_EXT:
        prefix += "ERROR:"
    if flags & vk.VK_DEBUG_REPORT_WARNING_BIT_EXT:
        prefix += "WARNING:"
    if flags & vk.VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
        prefix += "PERFORMANCE:"
    if flags & vk.VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
        prefix += "INFO:"
    if flags & vk.VK_DEBUG_REPORT_DEBUG_BIT_EXT:
        prefix += "DEBUG:"
    debugMessage = prefix + " [" + pLayerPrefix + "] Code " + str(msgCode) + " : " + pMsg
    if flags & vk.VK_DEBUG_REPORT_ERROR_BIT_EXT:
        print(debugMessage, file = sys.stderr)
    else:
        print(debugMessage)
    return vk.VK_FALSE

def setupDebugging(instance, flags, callback):
    CreateDebugReportCallback = vk.vkGetInstanceProcAddr(
        instance, "vkCreateDebugReportCallbackEXT")
    DestroyDebugReportCallback = vk.vkGetInstanceProcAddr(
        instance, "vkDestroyDebugReportCallbackEXT")
    dbgBreakCallback = vk.vkGetInstanceProcAddr(
        instance, "vkDebugReportMessageEXT")
    dbgCreateInfo = vk.VkDebugReportCallbackCreateInfoEXT(
        sType = vk.VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
        #flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
        flags = flags,
        pfnCallback = messageCallback)
    msgCallback = CreateDebugReportCallback(instance,dbgCreateInfo, None)
                # callback if (callBack != vk.VK_NULL_HANDLE) else msgCallback)

def freeDebugCallBack(instance):
    if msgCallback != vk.VK_NULL_HANDLE:
        DestroyDebugReportCallback(instance, msgCallback, None)
