CC = g++
TARGET = Tracker
SRCS = $(wildcard src/*.cpp)
OBJS = $(SRCS:.cpp=.o)
DLIBS = `pkg-config --libs opencv`
DINC = `pkg-config --cflags opencv`
$(TARGET):$(OBJS)
	$(CC) -o $@ $^ -std=c++11 -O3 $(DLIBS) 
clean:
	rm -rf $(TARGET) $(OBJS)  
%.o:%.cpp
	$(CC) -o $@ -c $< -std=c++11 -O3 $(DINC) 
