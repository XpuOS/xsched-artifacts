# BUILD_TYPE		= Debug / Release
BUILD_TYPE			= Release

# VERBOSE			= ON / OFF : enable verbose makefile
VERBOSE				= OFF

WORK_PATH			= $(shell pwd)
BUILD_PATH			= ${WORK_PATH}/build
OUTPUT_PATH			= ${WORK_PATH}/output

.PHONY: build
build: ${BUILD_PATH}/CMakeCache.txt
	rm -rf ${OUTPUT_PATH}; \
	cmake --build ${BUILD_PATH} --target install -- -j$(shell nproc)

${BUILD_PATH}/CMakeCache.txt:
	${MAKE} configure

.PHONY: configure
configure:
	cmake -B${BUILD_PATH}	\
		  -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}	\
		  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}		\
		  -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE}

.PHONY: clean
clean:
	@rm -rf ${BUILD_PATH} ${OUTPUT_PATH}
