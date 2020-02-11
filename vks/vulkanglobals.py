# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import platform

_WIN32 = (platform.system() == 'Windows')
VK_USE_PLATFORM_WIN32_KHR = _WIN32
VK_USE_PLATFORM_ANDROID_KHR = False
VK_USE_PLATFORM_WAYLAND_KHR = False
_DIRECT2DISPLAY = False
#VK_USE_PLATFORM_XCB_KHR = True
VK_USE_PLATFORM_XCB_KHR = not VK_USE_PLATFORM_WIN32_KHR

DEFAULT_FENCE_TIMEOUT = 100000000000
