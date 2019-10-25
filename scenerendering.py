# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import vulkan as vk
import vks.vulkanbuffer
import vks.vulkancamera
import vks.vulkanexamplebase
import vks.vulkanglobals
import vks.vulkantexture

import glm
import numpy as np
import imgui
import pyassimp
# no material.py in Fedora! copy by hand
import pyassimp.material

VERTEX_BUFFER_BIND_ID = 0

# From /usr/include/assimp/material.h
# pyassimp does not define them
AI_MATKEY_NAME = ('name', 0)
AI_MATKEY_COLOR_AMBIENT = ('ambient', 0)
AI_MATKEY_COLOR_DIFFUSE = ('diffuse', 0)
AI_MATKEY_COLOR_SPECULAR = ('specular', 0)
AI_MATKEY_OPACITY = ('opacity', 0)

class Scene:

    def __init__(self, vkdevice, queue):
        self.vulkanDevice = vkdevice
        self.queue = queue
        self.descriptorPool = None
        # We will be using separate descriptor sets (and bindings)
        # for material and scene related uniforms
        self.descriptorSetLayouts = {'material': None, 'scene': None}
        # We will be using one single index and vertex buffer
        # containing vertices and indices for all meshes in the scene
        # This allows us to keep memory allocations down
        self.vertexShape = np.dtype([('pos', np.float32, (3,)), ('normal', np.float32, (3,)), ('uv', np.float32, (2,)), ('color', np.float32, (3,))]) #position, normal, uv, color
        self.vertexBuffer = None
        self.indexBuffer = None

        self.descriptorSetScene = None
        self.aScene = None

        self.assetPath=""
        # Shader properites for a material
        # Will be passed to the shaders using push constant
        self.scenematerialShape = {'ambient': glm.vec4(0.0), 'diffuse': glm.vec4(0.0),'specular': glm.vec4(0.0), 'opacity': glm.vec1(0.0)}
        self.materials = []
        self.meshes = []
        # Shared ubo containing matrices used by all
        # materials and meshes
        self.uniformBuffer = None
        self.uniformData = {'projection': glm.mat4(), 'view': glm.mat4(), 'model': glm.mat4(), 'lightPos': glm.vec4(1.25, 8.35, 0.0, 0.0)}
        # Scene uses multiple pipelines
        self.pipelines = {'solid': None, 'blending': None, 'wireframe': None}
        # Shared pipeline layout
        self.pipelineLayout = None
        # For displaying only a single part of the scene
        self.renderSingleScenePart = False
        self.scenePartIndex = 0

        # TODO if that does not work, do not use createvksBuffer but do it by hand as in the demo
        uboSize = sum([glm.sizeof(ubo) for ubo in self.uniformData.values()])
        self.uniformBuffer = self.vulkanDevice.createvksBuffer(
            vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uboSize
        )
        self.uniformBuffer.map()

    def __del__(self):
        self.vertexBuffer.destroy()
        self.indexBuffer.destroy()
        for material in self.materials:
            material['diffuse'].destroy()
        vk.vkDestroyPipelineLayout(self.vulkanDevice.logicalDevice, self.pipelineLayout, None)
        vk.vkDestroyDescriptorSetLayout(self.vulkanDevice.logicalDevice, self.descriptorSetLayouts['material'], None)
        vk.vkDestroyDescriptorSetLayout(self.vulkanDevice.logicalDevice, self.descriptorSetLayouts['scene'], None)
        vk.vkDestroyDescriptorPool(self.vulkanDevice.logicalDevice, self.descriptorPool, None)
        vk.vkDestroyPipeline(self.vulkanDevice.logicalDevice, self.pipelines['solid'], None)
        vk.vkDestroyPipeline(self.vulkanDevice.logicalDevice, self.pipelines['blending'], None)
        vk.vkDestroyPipeline(self.vulkanDevice.logicalDevice, self.pipelines['wireframe'], None)
        self.uniformBuffer.destroy()

    # Get materials from the assimp scene and map to our scene structures
    def loadMaterials(self):
        for i in range(len(self.aScene.materials)):
            m = self.aScene.materials[i]
            scenematerial = {
                'name': None,
                'properties': {'ambient': glm.vec4(0.0), 'diffuse': glm.vec4(0.0),'specular': glm.vec4(0.0), 'opacity': glm.vec1(0.0)},
                'diffuse': vks.vulkantexture.Texture2D(),
                'descriptorSet': None,
                'pipeline': None
                }
            scenematerial['name'] = m.properties.get(AI_MATKEY_NAME)
            color = m.properties.get(AI_MATKEY_COLOR_AMBIENT) # returned as a list r, g, b, a hopefully
            scenematerial['properties']['ambient'] = glm.vec4(color) + glm.vec4(0.1)
            color = m.properties.get(AI_MATKEY_COLOR_DIFFUSE)
            scenematerial['properties']['diffuse'] = glm.vec4(color)
            color = m.properties.get(AI_MATKEY_COLOR_SPECULAR)
            scenematerial['properties']['specular'] = glm.vec4(color)
            if  m.properties.get(AI_MATKEY_OPACITY):
                scenematerial['properties']['opacity'] = glm.vec1(m.properties.get(AI_MATKEY_OPACITY))
            if scenematerial['properties']['opacity'] > glm.vec1(0.0):
                scenematerial['properties']['specular'] = glm.vec4(0.0)
            print("Material: \"" + scenematerial['name'] + "\"")

            # Textures
            texFormatSuffix = ""
            texFormat = None
            # Get supported compressed texture format
            if self.vulkanDevice.features.textureCompressionBC:
                texFormatSuffix = "_bc3_unorm"
                texFormat = vk.VK_FORMAT_BC3_UNORM_BLOCK
            elif self.vulkanDevice.features.textureCompressionASTC_LDR:
                texFormatSuffix = "_astc_8x8_unorm"
                texFormat = vk.VK_FORMAT_ASTC_8x8_UNORM_BLOCK
            elif self.vulkanDevice.features.textureCompressionETC2:
                texFormatSuffix = "_etc2_unorm"
                texFormat = vk.VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
            else:
                raise NotImplementedError("Device does not support any compressed texture format!")

            if m.properties.get(('file', pyassimp.material.aiTextureType_DIFFUSE)):
                texturefile = m.properties.get(('file', pyassimp.material.aiTextureType_DIFFUSE))
                print("  Diffuse: \"" + texturefile + "\"")
                texturefile = texturefile.replace("\\", "/")
                suffixidx = texturefile.find(".ktx")
                filename = texturefile[:suffixidx] + texFormatSuffix + texturefile[suffixidx:]
                scenematerial['diffuse'].loadFromFile(self.assetPath + filename, texFormat, self.vulkanDevice, self.queue)
            else:
                print("  Material has no diffuse, using dummy texture!")
                scenematerial['diffuse'].loadFromFile(self.assetPath + "dummy_rgba_unorm.ktx", vk.VK_FORMAT_R8G8B8A8_UNORM, self.vulkanDevice, self.queue)
            # For scenes with multiple textures per material we would need to check for additional texture types, e.g.:
            # aiTextureType_HEIGHT, aiTextureType_OPACITY, aiTextureType_SPECULAR, etc.

            # Assign pipeline
            scenematerial['pipeline'] = self.pipelines['solid'] if scenematerial['properties']['opacity'] == 0.0 else self.pipelines['blending']

            self.materials.append(scenematerial)

        # Generate descriptor sets for the materials
        # Descriptor pool
        poolSizes = []
        poolSize = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = len(self.materials)
        )
        poolSizes.append(poolSize)
        poolSize = vk.VkDescriptorPoolSize(
            type = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = len(self.materials)
        )
        poolSizes.append(poolSize)
        descriptorPoolInfo = vk.VkDescriptorPoolCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = len(poolSizes),
            pPoolSizes = poolSizes,
            maxSets = len(self.materials) + 1
        )
        self.descriptorPool = vk.vkCreateDescriptorPool(self.vulkanDevice.logicalDevice, descriptorPoolInfo, None)
        # Descriptor set and pipeline layouts
        # Set 0: Scene matrices
        setLayoutBindings = []
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_VERTEX_BIT,
            binding = 0
        )
        setLayoutBindings.append(layoutBinding)
        descriptorLayout = vk.VkDescriptorSetLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = len(setLayoutBindings),
            pBindings = setLayoutBindings
        )
        self.descriptorSetLayouts['scene'] = vk.vkCreateDescriptorSetLayout(self.vulkanDevice.logicalDevice, descriptorLayout, None)
        # Set 1: Material data
        setLayoutBindings = []
        layoutBinding = vk.VkDescriptorSetLayoutBinding(
            descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount = 1,
            stageFlags = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            binding = 0
        )
        setLayoutBindings.append(layoutBinding)
        descriptorLayout = vk.VkDescriptorSetLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = len(setLayoutBindings),
            pBindings = setLayoutBindings
        )
        self.descriptorSetLayouts['material'] = vk.vkCreateDescriptorSetLayout(self.vulkanDevice.logicalDevice, descriptorLayout, None)
        # Setup pipeline layout
        setLayouts = [self.descriptorSetLayouts['scene'], self.descriptorSetLayouts['material']]
        scenematerialSize = sum([glm.sizeof(scenematerialprop) for scenematerialprop in self.scenematerialShape.values()])
        pushConstantRange = vk.VkPushConstantRange(
            stageFlags = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            offset = 0,
            size = scenematerialSize
        )
        pPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = len(setLayouts),
            pSetLayouts = setLayouts,
            pushConstantRangeCount = 1,
            pPushConstantRanges = [ pushConstantRange ]
        )
        self.pipelineLayout = vk.vkCreatePipelineLayout(self.vulkanDevice.logicalDevice, pPipelineLayoutCreateInfo, None)
        # Material descriptor sets
        for m in self.materials:
            allocInfo = vk.VkDescriptorSetAllocateInfo(
                sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool = self.descriptorPool,
                descriptorSetCount = 1,
                pSetLayouts = [self.descriptorSetLayouts['material']]
            )
            descriptorSets = vk.vkAllocateDescriptorSets(self.vulkanDevice.logicalDevice, allocInfo)
            m['descriptorSet'] = descriptorSets[0]
            # Binding 0: Diffuse texture
            writeDescriptorSets = []
            writeDescriptorSet = vk.VkWriteDescriptorSet(
                sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet = m['descriptorSet'],
                descriptorType = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                dstBinding = 0,
                pImageInfo = [ m['diffuse'].descriptor ],
                descriptorCount = 1
            )
            writeDescriptorSets.append(writeDescriptorSet)
            vk.vkUpdateDescriptorSets(self.vulkanDevice.logicalDevice, len(writeDescriptorSets), writeDescriptorSets, 0, None)
        # Scene descriptor set
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = self.descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = [self.descriptorSetLayouts['scene']]
        )
        descriptorSets = vk.vkAllocateDescriptorSets(self.vulkanDevice.logicalDevice, allocInfo)
        self.descriptorSetScene = descriptorSets[0]
        writeDescriptorSets = []
        writeDescriptorSet = vk.VkWriteDescriptorSet(
            sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = self.descriptorSetScene,
            descriptorType = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            dstBinding = 0,
            pBufferInfo = [ self.uniformBuffer.descriptor ],
            descriptorCount = 1
        )
        writeDescriptorSets.append(writeDescriptorSet)
        vk.vkUpdateDescriptorSets(self.vulkanDevice.logicalDevice, len(writeDescriptorSets), writeDescriptorSets, 0, None)

    # Load all meshes from the scene and generate the buffers for rendering them
    def loadMeshes(self, copyCmd):
        # allocate numpy arrays
        vertexCount = 0
        indexCount = 0
        for aMesh in self.aScene.meshes:
            vertexCount += len(aMesh.vertices)
            indexCount += len(aMesh.faces) * 3
        vertices = np.empty((vertexCount,), dtype = self.vertexShape)
        indices = np.empty((indexCount,), dtype=np.uint32)
        indexBase = 0
        vertexCount = 0
        indexCount = 0
        for aMesh in self.aScene.meshes:
            print("Mesh \"" + aMesh.name +"\"")
            print("  Material: \"" + self.materials[aMesh.materialindex]["name"] + "\"")
            print("  Faces: " + str(len(aMesh.faces)))
            scenepart = {
                'material': self.materials[aMesh.materialindex],
                'indexBase': indexBase,
                'indexCount': len(aMesh.faces) * 3
            }
            self.meshes.append(scenepart)
            # Vertices
            hasUV = len(aMesh.texturecoords) > 0
            hasColor = len(aMesh.colors) > 0
            hasNormals = len(aMesh.normals) > 0
            print("  hasUV", hasUV, "hasColor", hasColor, "hasNormals", hasNormals)
            for v in range(len(aMesh.vertices)):
                vertices[vertexCount]['pos'] = aMesh.vertices[v]
                vertices[vertexCount]['pos'] = -vertices[vertexCount]['pos']
                vertices[vertexCount]['uv'] = aMesh.texturecoords[0][v][:2] if hasUV else [0.0, 0.0]
                vertices[vertexCount]['normal'] = aMesh.normals[v] if hasNormals else [0.0, 0.0, 0.0]
                vertices[vertexCount]['normal'] = -vertices[vertexCount]['normal']
                vertices[vertexCount]['color'] = aMesh.colors[v] if hasColor else [1.0, 1.0, 1.0]
                vertexCount += 1
            # Indices
            for f in range(len(aMesh.faces)):
                for j in range(3):
                    indices[indexCount] = aMesh.faces[f][j]
                    indexCount += 1
            indexBase += len(aMesh.faces) * 3
        # Create buffers
        # For better performance we only create one index and vertex buffer to keep number of memory allocations down
        vertexDataSize = vertices.size * vertices.itemsize
        indexDataSize = indices.size * indices.itemsize
        # Vertex buffer
        #  Staging buffer
        vertexStaging = self.vulkanDevice.createvksBuffer(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexDataSize, vertices)
        # Target
        self.vertexBuffer = self.vulkanDevice.createvksBuffer(vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexDataSize)
        # Index buffer
        #  Staging buffer
        indexStaging = self.vulkanDevice.createvksBuffer(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            indexDataSize, indices)
        # Target
        self.indexBuffer = self.vulkanDevice.createvksBuffer(vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexDataSize)
        # Copy
        cmdBufInfo = vk.VkCommandBufferBeginInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        )
        vk.vkBeginCommandBuffer(copyCmd, cmdBufInfo)
        copyRegion = vk.VkBufferCopy(size = vertexDataSize)
        vk.vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, self.vertexBuffer.buffer, 1, copyRegion)
        copyRegion = vk.VkBufferCopy(size = indexDataSize)
        vk.vkCmdCopyBuffer(copyCmd, indexStaging.buffer, self.indexBuffer.buffer, 1, copyRegion)
        vk.vkEndCommandBuffer(copyCmd)
        submitInfo = vk.VkSubmitInfo(
            sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pCommandBuffers = [ copyCmd ],
            commandBufferCount = 1
        )
        vk.vkQueueSubmit(self.queue, 1, submitInfo, vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        vertexStaging.destroy()
        indexStaging.destroy()

    def load(self, filename, copyCmd):
        flags = pyassimp.postprocess.aiProcess_PreTransformVertices | pyassimp.postprocess.aiProcess_Triangulate | pyassimp.postprocess.aiProcess_GenNormals
        try:
            self.aScene = pyassimp.load(filename, processing = flags)
        except pyassimp.AssimpError as e:
            print('Error parsing', filename, ':', e)
            raise RuntimeError
        if self.aScene is not None:
            self.loadMaterials()
            self.loadMeshes(copyCmd)
        else:
            print('Error parsing', filename)
            raise RuntimeError

    # Renders the scene into an active command buffer
    # In a real world application we would do some visibility culling in here
    def render(self, cmdBuffer, wireframe):
        offsets = [ 0 ]
        # Bind scene vertex and index buffers
        vk.vkCmdBindVertexBuffers(cmdBuffer, 0, 1, [ self.vertexBuffer.buffer ], offsets)
        vk.vkCmdBindIndexBuffer(cmdBuffer, self.indexBuffer.buffer, 0, vk.VK_INDEX_TYPE_UINT32)

        for i in range(len(self.meshes)):
            if self.renderSingleScenePart and i != self.scenePartIndex:
                continue
            # TODO : per material pipelines
            # vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, *mesh.material->pipeline);
            # We will be using multiple descriptor sets for rendering
            # In GLSL the selection is done via the set and binding keywords
            # VS: layout (set = 0, binding = 0) uniform UBO;
            # FS: layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;
            descriptorSets = []
            descriptorSets.append(self.descriptorSetScene)
            descriptorSets.append(self.meshes[i]['material']['descriptorSet'])
            vk.vkCmdBindPipeline(cmdBuffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipelines['wireframe'] if wireframe else self.meshes[i]['material']['pipeline'])
            vk.vkCmdBindDescriptorSets(cmdBuffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout, 0, len(descriptorSets), descriptorSets, 0, None)
            # Pass material properies via push constants
            propertiesSize = sum([glm.sizeof(pdata) for pdata in self.meshes[i]['material']['properties'].values()])
            propertiesData = np.concatenate((
                np.array(self.meshes[i]['material']['properties']['ambient']).flatten(order='C'),
                np.array(self.meshes[i]['material']['properties']['diffuse']).flatten(order='C'),
                np.array(self.meshes[i]['material']['properties']['specular']).flatten(order='C'),
                np.array(self.meshes[i]['material']['properties']['opacity']).flatten(order='C')
            ))
            #vk.vkCmdPushConstants(cmdBuffer, self.pipelineLayout, vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0,
			#	propertiesSize, propertiesData)
            # Render from the global scene vertex buffer using the mesh index offset
            vk.vkCmdDrawIndexed(cmdBuffer, self.meshes[i]['indexCount'], 1, 0, self.meshes[i]['indexBase'], 0)

class VulkanExample(vks.vulkanexamplebase.VulkanExampleBase):
    """
Summary:
* Renders a scene made of multiple parts with different materials and textures.
*
* The example loads a scene made up of multiple parts into one vertex and index buffer to only
* have one (big) memory allocation. In Vulkan it's advised to keep number of memory allocations
* down and try to allocate large blocks of memory at once instead of having many small allocations.
*
* Every part has a separate material and multiple descriptor sets (set = x layout qualifier in GLSL)
* are used to bind a uniform buffer with global matrices and the part's material's sampler at once.
*
* To demonstrate another way of passing data the example also uses push constants for passing
* material properties.
*
* Note that this example is just one way of rendering a scene made up of multiple parts in Vulkan.
    """
    def __init__(self):
        super().__init__(enableValidation=True)
        self.wireframe = False
        self.attachLight = False

        # also used in Scene
        self.vertexShape = np.dtype([('pos', np.float32, (3,)), ('normal', np.float32, (3,)), ('uv', np.float32, (2,)), ('color', np.float32, (3,))]) #position, normal, uv, color
        self.vertices = {'inputState': None, 'bindingDescriptions': [], 'attributeDescriptions': []}
        self.scene = None

        self.title = "Multi-part scene rendering"
        self.rotationSpeed = 0.5
        self.camera.type = vks.vulkancamera.CameraType.firstperson
        self.camera.movementSpeed = 7.5
        self.camera.position = glm.vec3(15.0, -13.5, 0.0)
        self.camera.setRotation(glm.vec3(5.0, 90.0, 0.0))
        self.camera.setPerspective(60.0, self.width / self.height, 0.1, 256.0)
        self.settings['overlay'] = True

    # Enable physical device features required for this example
    def getEnabledFeatures(self):
        # Fill mode non solid is required for wireframe display
        if self.deviceFeatures.fillModeNonSolid:
            self.enabledFeatures.fillModeNonSolid = vk.VK_TRUE
        if self.deviceFeatures.textureCompressionBC:
            self.enabledFeatures.textureCompressionBC = vk.VK_TRUE
        if self.deviceFeatures.textureCompressionETC2:
            self.enabledFeatures.textureCompressionETC2 = vk.VK_TRUE
        if self.deviceFeatures.textureCompressionASTC_LDR:
            self.enabledFeatures.textureCompressionASTC_LDR = vk.VK_TRUE

    def setupVertexDescriptions(self):
        vertexInputBinding = vk.VkVertexInputBindingDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            stride = self.vertexShape.itemsize,
            inputRate = vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        self.vertices['bindingDescriptions'].append(vertexInputBinding)
        # Location 0 : Position
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 0,
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['pos'][1] #  offsetof(vertexShape, pos)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        # Location 1 : Normal
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 1,
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['normal'][1] #  offsetof(vertexShape, normal)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        # Location 2 : Texture coordinates
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 2,
            format = vk.VK_FORMAT_R32G32_SFLOAT,
            offset = self.vertexShape.fields['uv'][1] #  offsetof(vertexShape, uv)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)
        # Location 3 : Color
        vertexInputAttribut = vk.VkVertexInputAttributeDescription(
            binding = VERTEX_BUFFER_BIND_ID,
            location = 3,
            format = vk.VK_FORMAT_R32G32B32_SFLOAT,
            offset = self.vertexShape.fields['color'][1] #  offsetof(vertexShape, color)
        )
        self.vertices['attributeDescriptions'].append(vertexInputAttribut)

        self.vertices['inputState'] = vk.VkPipelineVertexInputStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount = len(self.vertices['bindingDescriptions']),
            pVertexBindingDescriptions = self.vertices['bindingDescriptions'],
            vertexAttributeDescriptionCount = len(self.vertices['attributeDescriptions']),
            pVertexAttributeDescriptions = self.vertices['attributeDescriptions']
        )

    def updateUniformBuffers(self):
        if self.attachLight:
            self.scene.uniformData['lightPos'] = glm.vec4(-self.camera.position, 1.0)
        self.scene.uniformData['projection'] = self.camera.matrices['perspective']
        self.scene.uniformData['view'] = self.camera.matrices['view']
        self.scene.uniformData['model'] = glm.mat4(1.0)
        uDataSize = sum([glm.sizeof(udata) for udata in self.scene.uniformData.values()])
        uData = np.concatenate((
            np.array(self.scene.uniformData['projection']).flatten(order='C'),
            np.array(self.scene.uniformData['view']).flatten(order='C'),
            np.array(self.scene.uniformData['model']).flatten(order='C'),
            np.array(self.scene.uniformData['lightPos']).flatten(order='C')
        ))
        self.scene.uniformBuffer.copyTo(uData, uDataSize)

    def loadScene(self):
        copyCmd = super().createCommandBuffer(vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, False)
        self.scene = Scene(self.vulkanDevice, self.queue)
        self.scene.assetPath = self.getAssetPath() + "models/sibenik/"
        self.scene.load(self.getAssetPath() + "models/sibenik/sibenik.dae", copyCmd)
        vk.vkFreeCommandBuffers(self.device, self.cmdPool, 1, [ copyCmd ])
        self.updateUniformBuffers()

    def preparePipelines(self):
        inputAssemblyState = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology = vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            flags= 0,
            primitiveRestartEnable = vk.VK_FALSE
        )
        rasterizationState = vk.VkPipelineRasterizationStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = vk.VK_POLYGON_MODE_FILL,
            cullMode = vk.VK_CULL_MODE_BACK_BIT,
            frontFace = vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            flags = 0,
            depthClampEnable = vk.VK_FALSE,
            lineWidth = 1.0
        )
        blendAttachmentState = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask = 0xf,
            blendEnable = vk.VK_FALSE
        )
        colorBlendState = vk.VkPipelineColorBlendStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount = 1,
            pAttachments = [blendAttachmentState]
        )
        opState = vk.VkStencilOpState(
            #failOp = vk.VK_STENCIL_OP_KEEP,
            #passOp = vk.VK_STENCIL_OP_KEEP,
            compareOp = vk.VK_COMPARE_OP_ALWAYS
        )
        depthStencilState = vk.VkPipelineDepthStencilStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable = vk.VK_TRUE,
            depthWriteEnable = vk.VK_TRUE,
            depthCompareOp = vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            #depthBoundsTestEnable = vk.VK_FALSE,
            #stencilTestEnable = vk.VK_FALSE,
            front = opState,
            back = opState
        )
        viewportState = vk.VkPipelineViewportStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount = 1,
            scissorCount = 1,
            flags = 0
        )
        multisampleState = vk.VkPipelineMultisampleStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples = vk.VK_SAMPLE_COUNT_1_BIT,
            flags = 0
        )
        dynamicStateEnables = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        dynamicState = vk.VkPipelineDynamicStateCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount = len(dynamicStateEnables),
            pDynamicStates = dynamicStateEnables,
            flags = 0
        )
        shaderStages = []
        # Vertex shader
        shaderStage = vk.VkPipelineShaderStageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage = vk.VK_SHADER_STAGE_VERTEX_BIT,
            module = vks.vulkantools.loadShader(self.getAssetPath() + "shaders/scenerendering/scene.vert.spv", self.device),
            pName = "main"
        )
        shaderStages.append(shaderStage)
        # Fragment shader
        shaderStage = vk.VkPipelineShaderStageCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage = vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            module = vks.vulkantools.loadShader(self.getAssetPath() + "shaders/scenerendering/scene.frag.spv", self.device),
            pName = "main"
        )
        shaderStages.append(shaderStage)
        pipelineCreateInfo = vk.VkGraphicsPipelineCreateInfo(
            sType = vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            layout = self.scene.pipelineLayout,
            renderPass = self.renderPass,
            pVertexInputState = self.vertices['inputState'],
            pInputAssemblyState = inputAssemblyState,
            pRasterizationState = rasterizationState,
            pColorBlendState = colorBlendState,
            pMultisampleState = multisampleState,
            pViewportState = viewportState,
            pDepthStencilState = depthStencilState,
            pDynamicState = dynamicState,
            stageCount = len(shaderStages),
            pStages = shaderStages,
            flags = 0,
            basePipelineIndex = -1,
            basePipelineHandle = vk.VK_NULL_HANDLE
        )
        pipe = vk.vkCreateGraphicsPipelines(self.device, self.pipelineCache, 1, [pipelineCreateInfo], None)
        try:
            self.scene.pipelines['solid'] = pipe[0]
        except TypeError:
            self.scene.pipelines['solid'] = pipe
        # Alpha blended pipeline
        rasterizationState.cullMode = vk.VK_CULL_MODE_NONE
        blendAttachmentState.blendEnable = vk.VK_TRUE
        blendAttachmentState.colorBlendOp = vk.VK_BLEND_OP_ADD
        blendAttachmentState.srcColorBlendFactor = vk.VK_BLEND_FACTOR_SRC_COLOR
        blendAttachmentState.dstColorBlendFactor = vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR
        pipe = vk.vkCreateGraphicsPipelines(self.device, self.pipelineCache, 1, [pipelineCreateInfo], None)
        try:
            self.scene.pipelines['blending'] = pipe[0]
        except TypeError:
            self.scene.pipelines['blending'] = pipe
        # Wire frame rendering pipeline
        if self.deviceFeatures.fillModeNonSolid:
            rasterizationState.cullMode = vk.VK_CULL_MODE_BACK_BIT
            blendAttachmentState.blendEnable = vk.VK_FALSE
            rasterizationState.polygonMode = vk.VK_POLYGON_MODE_LINE
            rasterizationState.lineWidth = 1.0
            pipe = vk.vkCreateGraphicsPipelines(self.device, self.pipelineCache, 1, [pipelineCreateInfo], None)
            try:
                self.scene.pipelines['wireframe'] = pipe[0]
            except TypeError:
                self.scene.pipelines['wireframe'] = pipe

    def buildCommandBuffers(self):
        cmdBufInfo = vk.VkCommandBufferBeginInfo(
            sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        clearValues = []
        clearValue = vk.VkClearValue(
            color = self.defaultClearColor
        )
        clearValues.append(clearValue)
        clearValue = vk.VkClearValue(
            depthStencil = [1.0, 0 ]
        )
        clearValues.append(clearValue)
        offset = vk.VkOffset2D(x = 0, y = 0)
        extent = vk.VkExtent2D(width = self.width, height = self.height)
        renderArea = vk.VkRect2D(offset = offset, extent = extent)
        renderPassBeginInfo = vk.VkRenderPassBeginInfo(
            sType = vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext = None,
            renderPass = self.renderPass,
            renderArea = renderArea,
            clearValueCount = 2,
            pClearValues = clearValues,
        )
        for i in range(len(self.drawCmdBuffers)):
            renderPassBeginInfo.framebuffer = self.frameBuffers[i]
            vk.vkBeginCommandBuffer(self.drawCmdBuffers[i], cmdBufInfo)
            vk.vkCmdBeginRenderPass(self.drawCmdBuffers[i], renderPassBeginInfo, vk.VK_SUBPASS_CONTENTS_INLINE)
            viewport = vk.VkViewport(
                height = float(self.height),
                width = float(self.width),
                minDepth = 0.0,
                maxDepth = 1.0
            )
            vk.vkCmdSetViewport(self.drawCmdBuffers[i], 0, 1, [viewport])
            # Update dynamic scissor state
            offsetscissor = vk.VkOffset2D(x = 0, y = 0)
            extentscissor = vk.VkExtent2D(width = self.width, height = self.height)
            scissor = vk.VkRect2D(offset = offsetscissor, extent = extentscissor)
            vk.vkCmdSetScissor(self.drawCmdBuffers[i], 0, 1, [scissor])

            #self.scene.render(self.drawCmdBuffers[i], self.wireframe)
            self.scene.render(self.drawCmdBuffers[i], True)
            self.drawUI(self.drawCmdBuffers[i])
            vk.vkCmdEndRenderPass(self.drawCmdBuffers[i])
            vk.vkEndCommandBuffer(self.drawCmdBuffers[i])

    def prepare(self):
        super().prepare()
        self.setupVertexDescriptions()
        self.loadScene()
        self.preparePipelines()
        self.buildCommandBuffers()
        self.prepared = True

    def draw(self):
        super().prepareFrame()
        self.submitInfo.commandBufferCount = 1
        # TODO try to avoid creating submitInfo at each frame
        # need to get CData pointer on drawCmdBuffers[*]
        # self.submitInfo.pCommandBuffers[0] = self.drawCmdBuffers[self.currentBuffer]
        # vk.vkQueueSubmit(self.queue, 1, self.submitInfo, vk.VK_NULL_HANDLE)
        submitInfo = vk.VkSubmitInfo(
            sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pWaitDstStageMask = [ vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT ],
            pWaitSemaphores = [ self.semaphores['presentComplete'] ],
            waitSemaphoreCount = 1,
            signalSemaphoreCount = 1,
            pSignalSemaphores = [ self.semaphores['renderComplete'] ],
            pCommandBuffers = [ self.drawCmdBuffers[self.currentBuffer] ],
            commandBufferCount = 1
        )
        vk.vkQueueSubmit(self.queue, 1, submitInfo, vk.VK_NULL_HANDLE)
        super().submitFrame()

    def render(self):
        if not self.prepared:
            return
        #vk.vkDeviceWaitIdle(self.device)
        self.draw()

    def viewChanged(self):
        self.updateUniformBuffers()

scenerendering = VulkanExample()
scenerendering.main()
