import struct
import sys

class KtxFile:
    KTX11_FILE_IDENTIFIER = b'\xAB\x4B\x54\x58\x20\x31\x31\xBB\x0D\x0A\x1A\x0A'
    KTX_ENDIANNESS = 0x04030201

    def __init__(self, filename=None, mode = None):
        self.filename = None
        if filename is not None:
            self.filename = filename
        self.mode = mode
        self.ktx_file_identifier = None
        self.ktx_endianness = None
        self.ktx_gltype = None
        self.ktx_gltypesize = None
        self.ktx_glformat = None
        self.ktx_glinternal_format = None
        self.ktx_glbase_internal_format = None
        self.ktx_pixelwidth = None
        self.ktx_pixelheight = None
        self.ktx_pixeldepth = None
        self.ktx_numarrayelts = None
        self.ktx_numfaces = None
        self.ktx_nummimpmaplevels = None
        self.ktx_byteskeyvalue = None

        self.ktx_images = []
        self.ktx_keyvalues = dict()

        #self.endian = '<' if sys.byteorder == 'little' else '>'
        self.endian = '='
        self.fd = None
        self.offset_images = 0
        self.size = 0
        if self.filename is not None and self.mode == "r":
            self.open()
            self.readheader()
            self.parse_images()

    def open(self, filename=None, mode = None):
        if filename is not None and self.filename is not None and filename != self.filename:
            raise ValueError('KtxFile already has a filename: ' + self.filename)
        if self.filename is None and filename is None:
            raise ValueError('KtxFile has no filename')
        if self.filename is None and filename is not None:
            self.filename = filename
        if self.fd is not None:
            raise RuntimeError('KtxFile '+self.filename+' is already opened')
        if mode is not None:
            self.mode = mode
        if self.mode == 'w':
            raise NotImplementedError('KtxFile: write operation is not implemented')
        self.fd = open(self.filename, 'rb')

    def readheader(self):
        self.ktx_file_identifier = self.fd.read(12)
        if self.ktx_file_identifier != KtxFile.KTX11_FILE_IDENTIFIER:
            raise RuntimeError('KtFile '+ self.filename + ': Bad header (' + self.ktx_file_identifier + ')')
        self.ktx_endianness,  = struct.unpack('=I', self.fd.read(4))
        if self.ktx_endianness != KtxFile.KTX_ENDIANNESS:
            self.endian = '>' if sys.byteorder == 'little' else '<'
        self.ktx_gltype,  = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_gltypesize,  = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_glformat, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_glinternal_format, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_glbase_internal_format, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_pixelwidth, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_pixelheight, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_pixeldepth, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_numarrayelts, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_numfaces, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_nummimpmaplevels, = struct.unpack(self.endian+'I', self.fd.read(4))
        self.ktx_byteskeyvalue, = struct.unpack(self.endian+'I', self.fd.read(4))
        readkeyvalues = 0
        while readkeyvalues < self.ktx_byteskeyvalue:
            keyvaluesize, = struct.unpack(self.endian+'I', self.fd.read(4))
            keyvalue = self.fd.read(keyvaluesize)
            try:
                keyvaluebytes = keyvalue.split(b'\x00')
                key = keyvaluebytes[0]
                value = keyvaluebytes[1]
                self.ktx_keyvalues[key] = value
                valuePadding = self.fd.read(3 - ((keyvaluesize + 3) % 4))
            except:
                print('can not read key/value')
            readkeyvalues += (4 + keyvaluesize + (3 - ((keyvaluesize + 3) % 4)))
        self.offset_images = self.fd.tell()

    def parse_images(self):
        if self.ktx_numfaces == 6 and self.ktx_numarrayelts == 0:
            raise NotImplementedError('Non-array cubemap textures not implemented')
        if self.offset_images == 0:
            self.fd.seek(0, 0)
            self.readheader()
        if self.fd.tell() != self.offset_images:
            self.fd.seek(self.offset_images, 0)
        self.size = 0
        offset = self.offset_images
        imgsize = self.fd.read(4)
        while imgsize != b'':
            imgsize, =  struct.unpack(self.endian+'I', imgsize)
            self.size += imgsize
            offset += 4
            image = {'size': imgsize, 'offset': offset}
            self.ktx_images.append(image)
            self.fd.seek(imgsize, 1)
            offset += imgsize
            # beware of mip padding
            imgsize = self.fd.read(4)

    def get_image(self, miplevel = 0, arraylayer = 0):
        if self.ktx_numarrayelts != 0 and arraylayer >= self.ktx_numarrayelts:
            raise ValueError('no such many layers in texture (' + str(arraylayer) + ' requested / ' + str(self.ktx_numarrayelts) + ' present)')
        if miplevel >= self.ktx_nummimpmaplevels:
            raise ValueError('no such many miplevels in texture (' + str(miplevel) + ' requested / ' + str(self.ktx_nummimpmaplevels) + ' present)')
        if self.ktx_numarrayelts != 0:
            raise NotImplementedError('Array textures not implemented')
        self.fd.seek(self.ktx_images[miplevel]['offset'], 0)
        return self.fd.read(self.ktx_images[miplevel]['size'])

if __name__ == '__main__':
    import ktxfile
    #f = ktxfile.KtxFile('Vulkan/data/textures/metalplate01_rgba.ktx', "r")
    f = ktxfile.KtxFile('Vulkan/data/textures/rocks_color_astc_8x8_unorm.ktx', "r")
    f.parse_images()
    # view image if uncompressed RGBA
    # im=f.get_image(4)
    # from PIL import Image
    # pim = Image.frombytes('RGBA', (32,32), im) # 32 = 2^(10-(4+1))
    # pim.show()
    # if compressed astc
    # save one mip level as ASTC file for decoding
    im0 = f.get_image(0)
    with open('output.astc', 'wb') as o:
        o.write(struct.pack('I', 0x5CA1AB13))
        # this info may be deduced from gl_*_format/vk_format mapping
        # see libktx https://github.com/KhronosGroup/KTX-Software/blob/master/lib/vk_format.h
        o.write(b'\x08')
        o.write(b'\x08')
        o.write(b'\x01')
        o.write(struct.pack('I', f.ktx_pixelwidth)[0:3])
        o.write(struct.pack('I', f.ktx_pixelheight)[0:3])
        o.write(struct.pack('I', 1 if f.ktx_pixeldepth == 0 else f.ktx_pixeldepth)[0:3])
        o.write(im0)
    # decode with astcenc (from mali arm texture compression tool mali-tct)
    # mali-tct]$ bin/astcenc -d output.astc rock_colors.tga

# for texture format compression see https://developer.nvidia.com/astc-texture-compression-for-game-assets
# for decoding information see bimg Image src https://github.com/bkaradzic/bimg/blob/master/src/image.cpp

# ASTC specific info: http://malideveloper.arm.com/downloads/Stacy_ASTC_white%20paper.pdf
# ASTC Specification https://www.khronos.org/registry/DataFormat/specs/1.2/dataformat.1.2.html#ASTC
# source code for astcenc https://github.com/ARM-software/astc-encoder
# header used by astcenc (decoder/encoder from ARM/Mali )
# struct astc_header
# {
#     uint8_t magic [ 4 ];
#     uint8_t blockdim_x;
#     uint8_t blockdim_y;
#     uint8_t blockdim_z ;
#     uint8_t xsize [ 3 ];
#     uint8_t ysize [ 3 ];
#     uint8_t zsize [ 3 ];
# };
