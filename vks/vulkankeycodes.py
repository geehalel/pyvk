# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import enum

from vks.vulkanglobals import  *
if VK_USE_PLATFORM_XCB_KHR:
    class Keycode(enum.IntEnum):
        KEY_ESCAPE = 0x9
        KEY_F1 = 0x43
        KEY_F2 = 0x44
        KEY_F3 = 0x45
        KEY_F4 = 0x46
        KEY_W = 0x19
        KEY_A = 0x26
        KEY_S = 0x27
        KEY_D = 0x28
        KEY_P = 0x21
        KEY_SPACE = 0x41
        KEY_KPADD = 0x56
        KEY_KPSUB = 0x52
        KEY_B = 0x38
        KEY_F = 0x29
        KEY_L = 0x2E
        KEY_N = 0x39
        KEY_O = 0x20
        KEY_T = 0x1C
    
if VK_USE_PLATFORM_WIN32_KHR:
    import win32con
    class Keycode(enum.IntEnum):
        KEY_ESCAPE = win32con.VK_ESCAPE
        KEY_F1 = win32con.VK_F1
        KEY_F2 = win32con.VK_F2
        KEY_F3 = win32con.VK_F3
        KEY_F4 = win32con.VK_F4
        KEY_W = 0x57
        KEY_A = 0x41
        KEY_S = 0x53
        KEY_D = 0x44
        KEY_P = 0x50
        KEY_SPACE = 0x20
        KEY_KPADD = 0x6B
        KEY_KPSUB = 0x6D
        KEY_B = 0x42
        KEY_F = 0x46
        KEY_L = 0x4C
        KEY_N = 0x4E
        KEY_O = 0x4F
        KEY_T = 0x54