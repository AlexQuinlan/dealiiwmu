# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/quinlan/Github/dealii/examples/cadex

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/quinlan/Github/dealii/examples/cadex

# Include any dependencies generated for this target.
include CMakeFiles/cadex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cadex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cadex.dir/flags.make

CMakeFiles/cadex.dir/cadex.cc.o: CMakeFiles/cadex.dir/flags.make
CMakeFiles/cadex.dir/cadex.cc.o: cadex.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/quinlan/Github/dealii/examples/cadex/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cadex.dir/cadex.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cadex.dir/cadex.cc.o -c /home/quinlan/Github/dealii/examples/cadex/cadex.cc

CMakeFiles/cadex.dir/cadex.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cadex.dir/cadex.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/quinlan/Github/dealii/examples/cadex/cadex.cc > CMakeFiles/cadex.dir/cadex.cc.i

CMakeFiles/cadex.dir/cadex.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cadex.dir/cadex.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/quinlan/Github/dealii/examples/cadex/cadex.cc -o CMakeFiles/cadex.dir/cadex.cc.s

# Object files for target cadex
cadex_OBJECTS = \
"CMakeFiles/cadex.dir/cadex.cc.o"

# External object files for target cadex
cadex_EXTERNAL_OBJECTS =

cadex: CMakeFiles/cadex.dir/cadex.cc.o
cadex: CMakeFiles/cadex.dir/build.make
cadex: /home/quinlan/local/dealii/debug/petsc/lib/libdeal_II.g.so.9.5.0-pre
cadex: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
cadex: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
cadex: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
cadex: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
cadex: /usr/lib/x86_64-linux-gnu/openmpi/lib/libopen-pal.so
cadex: /usr/lib/x86_64-linux-gnu/libtbb.so
cadex: /usr/lib/x86_64-linux-gnu/libz.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_system.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_thread.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_regex.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
cadex: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
cadex: /usr/lib/x86_64-linux-gnu/libumfpack.so
cadex: /usr/lib/x86_64-linux-gnu/libcholmod.so
cadex: /usr/lib/x86_64-linux-gnu/libccolamd.so
cadex: /usr/lib/x86_64-linux-gnu/libcolamd.so
cadex: /usr/lib/x86_64-linux-gnu/libcamd.so
cadex: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
cadex: /usr/lib/x86_64-linux-gnu/libamd.so
cadex: /usr/lib/x86_64-linux-gnu/libpetsc.so
cadex: /usr/lib/x86_64-linux-gnu/libarpack.so
cadex: /usr/lib/x86_64-linux-gnu/libmkl_gf_lp64.so
cadex: /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so
cadex: /usr/lib/x86_64-linux-gnu/libmkl_core.so
cadex: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
cadex: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
cadex: /home/quinlan/.local/bin/lib/libTKBO.so
cadex: /home/quinlan/.local/bin/lib/libTKBool.so
cadex: /home/quinlan/.local/bin/lib/libTKBRep.so
cadex: /home/quinlan/.local/bin/lib/libTKernel.so
cadex: /home/quinlan/.local/bin/lib/libTKFeat.so
cadex: /home/quinlan/.local/bin/lib/libTKFillet.so
cadex: /home/quinlan/.local/bin/lib/libTKG2d.so
cadex: /home/quinlan/.local/bin/lib/libTKG3d.so
cadex: /home/quinlan/.local/bin/lib/libTKGeomAlgo.so
cadex: /home/quinlan/.local/bin/lib/libTKGeomBase.so
cadex: /home/quinlan/.local/bin/lib/libTKHLR.so
cadex: /home/quinlan/.local/bin/lib/libTKIGES.so
cadex: /home/quinlan/.local/bin/lib/libTKMath.so
cadex: /home/quinlan/.local/bin/lib/libTKMesh.so
cadex: /home/quinlan/.local/bin/lib/libTKOffset.so
cadex: /home/quinlan/.local/bin/lib/libTKPrim.so
cadex: /home/quinlan/.local/bin/lib/libTKShHealing.so
cadex: /home/quinlan/.local/bin/lib/libTKSTEP.so
cadex: /home/quinlan/.local/bin/lib/libTKSTEPAttr.so
cadex: /home/quinlan/.local/bin/lib/libTKSTEPBase.so
cadex: /home/quinlan/.local/bin/lib/libTKSTEP209.so
cadex: /home/quinlan/.local/bin/lib/libTKSTL.so
cadex: /home/quinlan/.local/bin/lib/libTKTopAlgo.so
cadex: /home/quinlan/.local/bin/lib/libTKXSBase.so
cadex: CMakeFiles/cadex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/quinlan/Github/dealii/examples/cadex/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cadex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cadex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cadex.dir/build: cadex

.PHONY : CMakeFiles/cadex.dir/build

CMakeFiles/cadex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cadex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cadex.dir/clean

CMakeFiles/cadex.dir/depend:
	cd /home/quinlan/Github/dealii/examples/cadex && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/quinlan/Github/dealii/examples/cadex /home/quinlan/Github/dealii/examples/cadex /home/quinlan/Github/dealii/examples/cadex /home/quinlan/Github/dealii/examples/cadex /home/quinlan/Github/dealii/examples/cadex/CMakeFiles/cadex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cadex.dir/depend

