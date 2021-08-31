# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import numpy as np
import cffi
ffi=cffi.FFI()
class DataWrapper:
    def __init__(self):
        self.__array_interface__ = {'shape': (1,), 'typestr': '|u1', 'version': 3}

class Buffer:
    def __init__(self):
        self.device = None
        self.buffer = vk.VK_NULL_HANDLE
        self.memory = vk.VK_NULL_HANDLE
        self.descriptor = vk.VkDescriptorBufferInfo()
        self.size = 0
        self.alignment = 0
        self.mapped = None
        self.usageFlags = None
        self.memoryPropertyFlags = None

    def map(self, size = vk.VK_WHOLE_SIZE, offset = 0):
        """
Map a memory range of this buffer. If successful, mapped points to the specified buffer range.
@param size (Optional) Size of the memory range to map. Pass VK_WHOLE_SIZE to map the complete buffer range.
@param offset (Optional) Byte offset from beginning

@return VkResult of the buffer mapping call
        """
        if size == vk.VK_WHOLE_SIZE:
            size = self.size
        self.mapped = vk.vkMapMemory(self.device, self.memory, offset, size, 0)

    def unmap(self):
        """
Unmap a mapped memory range
@note Does not return a result as vkUnmapMemory can't fail
        """
        if self.mapped is not None:
            vk.vkUnmapMemory(self.device, self.memory)
            self.mapped = None

    def bind(self, offset = 0):
        """
Attach the allocated memory block to the buffer

 @param offset (Optional) Byte offset (from the beginning) for the memory region to bind

 @return VkResult of the bindBufferMemory call
        """
        vk.vkBindBufferMemory(self.device, self.buffer, self.memory, offset)

    def setupDescriptor(self, size = vk.VK_WHOLE_SIZE, offset = 0):
        """
Setup the default descriptor for this buffer

@param size (Optional) Size of the memory range of the descriptor
@param offset (Optional) Byte offset from beginning
        """
        if size == vk.VK_WHOLE_SIZE:
            size = self.size
        self.descriptor.offset = offset
        self.descriptor.buffer = self.buffer
        self.descriptor.range = size

    def copyTo(self, data, size):
        """
Copies the specified data to the mapped buffer

@param data Pointer to the data to copy
@param size Size of the data to copy in machine units
        """
        # TODO find another way to memcpy
        assert(self.mapped is not None)
        #memorywrapper = np.array(self.mapped, copy=False)
        ##databuf = np.resize(np.frombuffer(data, dtype=np.uint8), size)
        ##databuf = np.frombuffer(data, dtype = np.uint8)
        #databuf = DataWrapper()
        #databuf.__array_interface__['shape'] = (size,)
        ##databuf.__array_interface__['data'] = (id(data), False)
        #databuf.__array_interface__['data'] = data
        #np.copyto(memorywrapper[:size], databuf, casting='no')
        ffi.memmove(self.mapped, data, size)
        

    def flush(self, size = vk.VK_WHOLE_SIZE, offset = 0):
        """
Flush a memory range of the buffer to make it visible to the device

@note Only required for non-coherent memory

@param size (Optional) Size of the memory range to flush. Pass VK_WHOLE_SIZE to flush the complete buffer range.
@param offset (Optional) Byte offset from beginning

@return VkResult of the flush call
        """
        if size == vk.VK_WHOLE_SIZE:
            size = self.size
        mappedRange = vk.VkMappedMemoryRange(
            sType = vk.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            memory = self.memory,
            offset = offset,
            size = size
        )
        vk.vkFlushMappedMemoryRanges(self.device, 1, mappedRange)

    def invalidate(self, size = vk.VK_WHOLE_SIZE, offset = 0):
        """
Invalidate a memory range of the buffer to make it visible to the host

 @note Only required for non-coherent memory

@param size (Optional) Size of the memory range to invalidate. Pass VK_WHOLE_SIZE to invalidate the complete buffer range.
 @param offset (Optional) Byte offset from beginning

@return VkResult of the invalidate call
        """
        mappedRange = vk.VkMappedMemoryRange(
            sType = vk.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            memory = self.memory,
            offset = offset,
            size = size
        )
        vk.vkInvalidateMappedMemoryRanges(self.device, 1, mappedRange)

    def destroy(self):
        """
 Release all Vulkan resources held by this buffer
        """
        if self.buffer:
            vk.vkDestroyBuffer(self.device, self.buffer, None)
        if self.memory:
            vk.vkFreeMemory(self.device, self.memory, None)
