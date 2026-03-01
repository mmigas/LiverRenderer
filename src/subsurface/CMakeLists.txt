set(MI_PLUGIN_PREFIX "subsurface")

#add_plugin(dipole dipole.cpp irrproc.h irrproc.cpp irrtree.cpp irrtree.h bluenoise.hpp)
add_plugin(vaescatter vaescatter.cpp)
set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
