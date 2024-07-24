#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#define SHARED_MEMORY_NAME "/my_shared_memory"
#define SHARED_MEMORY_SIZE (sizeof(SharedMemory))

#define SENDER_SEMAPHORE_NAME "/my_sender_semaphore"
#define RECEIVER_SEMAPHORE_NAME "/my_receiver_semaphore"

#define TPU_IMG_SIZE 224*224*3


struct SharedMemory {
    float result;
    char image[TPU_IMG_SIZE];
};

#endif
