CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -g

CORE_SRCS = src/core/card.cpp src/core/deck.cpp src/core/player.cpp src/core/hokm.cpp
TEST_SRCS = tests/test_core.cpp

all: test_core

test_core: $(CORE_SRCS) $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) -o test_core $(CORE_SRCS) $(TEST_SRCS)

clean:
	rm -f test_core
