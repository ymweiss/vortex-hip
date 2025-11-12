#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "vortex_hip::vortex_hip" for configuration "Release"
set_property(TARGET vortex_hip::vortex_hip APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(vortex_hip::vortex_hip PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhip_vortex.so.1.0.0"
  IMPORTED_SONAME_RELEASE "libhip_vortex.so.1"
  )

list(APPEND _cmake_import_check_targets vortex_hip::vortex_hip )
list(APPEND _cmake_import_check_files_for_vortex_hip::vortex_hip "${_IMPORT_PREFIX}/lib/libhip_vortex.so.1.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
