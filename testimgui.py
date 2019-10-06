import imgui

imgui.create_context()

io=imgui.get_io()
texWidth, texHeight, fontData = io.fonts.get_tex_data_as_rgba32()

io.display_size = imgui.Vec2(100,200)
imgui.new_frame()
imgui.render()
imgui.new_frame()
imgui.begin('Vulkan Example', None, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_RESIZE| imgui.WINDOW_NO_MOVE)
imgui.text_unformatted('yeah!!!')
imgui.text('yeah!!!')
imgui.text('yeah!!!')
imgui.end()
imgui.render()
imDrawData = imgui.get_draw_data()

# repeat ?
imgui.new_frame()
imgui.render()
imgui.new_frame()
imgui.begin('Vulkan Example', None, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_RESIZE| imgui.WINDOW_NO_MOVE)
imgui.text_unformatted('yeah!!!')
imgui.text('yeah!!!')
imgui.text('yeah!!!')
imgui.end()
imgui.render()
imDrawData = imgui.get_draw_data()
cmd_list=imDrawData.commands_lists[0]
pcmd=cmd_list.commands[0]
