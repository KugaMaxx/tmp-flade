# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build

# Include any dependencies generated for this target.
include CMakeFiles/flame_scout.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/flame_scout.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/flame_scout.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/flame_scout.dir/flags.make

CMakeFiles/flame_scout.dir/flame_scout.cpp.o: CMakeFiles/flame_scout.dir/flags.make
CMakeFiles/flame_scout.dir/flame_scout.cpp.o: ../flame_scout.cpp
CMakeFiles/flame_scout.dir/flame_scout.cpp.o: CMakeFiles/flame_scout.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/flame_scout.dir/flame_scout.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/flame_scout.dir/flame_scout.cpp.o -MF CMakeFiles/flame_scout.dir/flame_scout.cpp.o.d -o CMakeFiles/flame_scout.dir/flame_scout.cpp.o -c /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/flame_scout.cpp

CMakeFiles/flame_scout.dir/flame_scout.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flame_scout.dir/flame_scout.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/flame_scout.cpp > CMakeFiles/flame_scout.dir/flame_scout.cpp.i

CMakeFiles/flame_scout.dir/flame_scout.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flame_scout.dir/flame_scout.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/flame_scout.cpp -o CMakeFiles/flame_scout.dir/flame_scout.cpp.s

# Object files for target flame_scout
flame_scout_OBJECTS = \
"CMakeFiles/flame_scout.dir/flame_scout.cpp.o"

# External object files for target flame_scout
flame_scout_EXTERNAL_OBJECTS =

../flame_scout.cpython-38-x86_64-linux-gnu.so: CMakeFiles/flame_scout.dir/flame_scout.cpp.o
../flame_scout.cpython-38-x86_64-linux-gnu.so: CMakeFiles/flame_scout.dir/build.make
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
../flame_scout.cpython-38-x86_64-linux-gnu.so: CMakeFiles/flame_scout.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../flame_scout.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flame_scout.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/flame_scout.cpython-38-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/flame_scout.dir/build: ../flame_scout.cpython-38-x86_64-linux-gnu.so
.PHONY : CMakeFiles/flame_scout.dir/build

CMakeFiles/flame_scout.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/flame_scout.dir/cmake_clean.cmake
.PHONY : CMakeFiles/flame_scout.dir/clean

CMakeFiles/flame_scout.dir/depend:
	cd /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build /home/dszh/Workspace/tmp-flade/samples/bec_svm/generator/build/CMakeFiles/flame_scout.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/flame_scout.dir/depend

