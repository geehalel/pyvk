# pyvk : Python Vulkan examples and demos

`pyvk` is a Python port of the wonderful framework from [Sascha Willems vulkan examples](https://github.com/SaschaWillems/Vulkan).
It has been started as a mean for me to discover Vulkan. The initial objectives are:
- use some of the proposed classes to ease development (but some classes require stuff not ported to python (gli?))
- part of these classes are in a `vks` namespace, thus make a `vks` python module
- port the main class (`VulkanExampleBase`) to Python
- try to implement `triangle` and `scenerendering`

### Current status
Essential classes have been ported to Python in a `vks` Pyhton module, and the `triangle` example is running.
Moreover `imgui` integration is also effective, it is also now included in the `triangle` demo. However not every methods have yet been rewritten in those classes. For instance there is no benchmarking facilities.

Only the XCB platform binding has been made, so this only runs on `linux`. It has been tested under Fedora 30 and Ubuntu 18.04, using  both IGP Intel drivers and nvidia proprietary drivers.

###### *FPS comparisons with the C++ original examples*
These are Frames Per Second values computed in the program itself.
They are read directly from the [screenshots](#screenshots). All FPS values here are stable. Min/Max frame times will also be added.

| Example | CPU-Card-Driver-platform-os | C++ | Python |
|---------|-----|--------|
| triangle | i3 3220T-IGP-Intel Ivybridge-XCB-linux |798 | 757 |
| triangle (imgui) | i3 3220T-IGP-Intel Ivybridge-XCB-linux |637 | 427 |

### Running the example(s)
*First install the requirements (see [below](#installation-requirements)).
At the moment I also use [numpy](https://numpy.org/) as a mean to access C arrays from Python, thus you should also install it (`pip3 install --user numpy`).*

Simply clone the repository in a directory of your choice and change directory
```
git clone https://github.com/geehalel/pyvk.git
cd pyvk
```
then run the desired demo
```
python3 triangle.py
```

### Installation requirements

- install [python vulkan binding](https://github.com/realitix/vulkan)
```
pip3 install vulkan --user
```
- install [python xcb binding](https://github.com/tych0/xcffib)
```
pip3 install xcffib --user
```
- install [python glm binding](https://github.com/Zuzu-Typ/PyGLM)
```
pip3 install pyglm --user  ```
- install [python imgui binding](https://github.com/swistakm/pyimgui)
```
pip3 install imgui --user
```
- install [python wayland binding](https://github.com/flacjacket/pywayland) (not useful for now)
```
pip3 install pywayland --user
```

### Screenshots

These are comparative screenshots of the original examples written in C++ and the ones here written in Python.

|Example | C++ | Python |
|--------|-----|--------|
|triangle | ![](figs/triangle_cpp.png) | ![](figs/triangle_python.png) |
|triangle (imgui) | ![](figs/triangle_imgui_cpp.png) | ![](figs/triangle_imgui_python.png) |
