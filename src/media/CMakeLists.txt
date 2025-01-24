set(MI_PLUGIN_PREFIX "media")

add_plugin(homogeneous homogeneous.cpp)
add_plugin(heterogeneous heterogeneous.cpp)
add_plugin(parenchyma organic_material.h parenchyma.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
