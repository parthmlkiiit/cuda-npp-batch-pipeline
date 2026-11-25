# Build for Linux / the Coursera lab (CUDA toolkit with NPP).
#   make            build bin/npp_batch
#   make run        build + run on data/sipi (same as ./run.sh without packing)
#   make data       download and convert the SIPI images
#   make lint       cpplint (Google C++ style) on src/
#   make format     clang-format -i on src/
#   make clean

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc

CXXFLAGS  := -std=c++17 -O2
INCLUDES  := -I. -isystem third_party -I$(CUDA_PATH)/include
NPP_LIBS  := -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppist \
             -lnppisu
LDFLAGS   := -L$(CUDA_PATH)/lib64 $(NPP_LIBS) -lcudart

SRCS      := $(wildcard src/*.cc)
OBJS      := $(patsubst src/%.cc,build/%.o,$(SRCS))
TARGET    := bin/npp_batch

RUN_ARGS  ?= --input_dir=data/sipi --output_dir=output

.PHONY: all run data lint format clean

all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(NVCC) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/%.o: src/%.cc src/*.h | build
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

bin build:
	mkdir -p $@

run: $(TARGET)
	./$(TARGET) $(RUN_ARGS)

data:
	./scripts/download_data.sh

lint:
	cpplint --recursive src

format:
	clang-format -i src/*.cc src/*.h

clean:
	rm -rf build bin
