# 04 Pthreads
## 1. 进程、线程和 Pthreads
1. 进程：运行程序的实例
2. 线程：轻量级进程，一个进程中有多个控制线程
3. 运用多线程编程技术，有助于我们编写并行程序

## 2. Pthreads 基础知识
1. Hello World!
   ```c
   #include <stdio.h>
   #include <stdlib.h>
   #include <pthread.h>

   int thread_count; // number of threads

   void* hello(void* rank) {
      long my_rank = (long) rank; 

      printf("hello from the thread %ld of %d\n", my_rank, thread_count); 

      return NULL; 
   }

   int main(int argc, char* argv[]) {
      long thread; 
      pthread_t* thread_handles; 
      thread_count = strtol(argv[1], NULL, 10); 

      // get number of threads from command line
      thread_handles = malloc(thread_count * sizeof(pthread_t));

      // generate threads 
      for (thread = 0; thread < thread_count; thread++) {
         pthread_create(&thread_handles[thread], NULL, hello, (void*) thread); 
      }

      printf("hello from the main thread\n"); 

      // join threads
      for (thread = 0; thread < thread_count; thread++) {
         pthread_join(thread_handles[thread], NULL); 
      }

      free(thread_handles); 
      return 0; 
   }
   ```
2. 执行
   ``` shell
   gcc -o pth_hello pth_hello.c -lpthread
   ./pth_hello <number of threads>
   ```
3. `pthread_create`: 创建线程，第一个指针指向对应的 `pthread_t` 对象，第二个参数直接 `NULL`；第三个是要执行的函数，第四个指向传给函数的参数
   ```c
   int pthread_create (
      pthread_t*            thread_p,       // out
      const pthread_attr_t* attr_p,         // in
      void*                 (*func)(void*), // in 
      void*                 arg_p           // in
   ); 
   ```
4. `pthread_join`: 停止线程，调用一次 `pthread_join` 将等待 pthread_t 对象所关联的那个线程结束；第二个参数可以接收任意由 pthread_t 对象所关联的那个线程产生的返回值
   ```c
   int pthread_join (
      pthread_t*            thread_p,       // in
      void**                ret_val_p       // out
   );
   ```

## 3. Pthread 矩阵向量乘法
对于 `Ax = y` 基本思想就是将矩阵 `A` 按行划分为等分的多个子矩阵（假设矩阵行数可以整除设定线程数量）然后每个线程根据自己的 rank 将子矩阵与 `x` 相乘得到对应部分的 `y` 的值 ![pthread_matrix_vector](gallery/pthread_matrix_vector.png)
```c
# define PTHREAD_CNT 8

void* pth_matrix_vector(void* rank) {
   long my_rank = (long)rank; 
   int local_m = m / PTHREAD_CNT; 
   int my_first_row = my_rank * local_m; 
   int my_last_row = (my_rank + 1) * local_m; 

   for (int i = my_first_row; i < my_last_row; i++) {
      y[i] = 0.0; 
      for (int j = 0; j < n; j++) {
         y[i] += A[i][j] * x[j]; 
      }
   }

   return NULL; 
}
```

## 4. 互斥与同步
1. 临界区：是指一段代码，其中包含对共享资源的访问或修改操作。这些共享资源可以是内存中的变量、文件、设备等；一般一次只允许一个线程执行该代码片段（即进入临界区）
2. 竞争条件：当多个线程访问临界区的时候，可能会发生修改共享变量，此时不同的访问顺序可能导致不同的结果，因此产生竞争条件
3. 忙等待：线程不停地测试某个条件满足以进入临界区，但是实际上直到条件满足之前的测试操作都是徒劳的；忙等待不是保护临界区的唯一方法。事实上，还有很多更好的方法。然而，因为临界区中的代码一次只能由一个线程运行，所以无论如何限制访问临界区，都必须串行地执行其中的代码。如果可能的话，我们应该最小化执行临界区的次数。能够大幅度提高性能的一个方法是：给每个线程配置私有变量来存储各自的部分和，然后用 `for` 循环一次性将所有部分和加在一起算出总和
4. 互斥量：即互斥锁，互斥量可以用来限制每次只有一个线程能进入临界区。互斥量保证了一个线程独享临界区；Pthreads 提供了互斥锁的实现 `pthread_mutex_t` 
   1. `int pthread_mutex_init(pthread_mutex_t* mutex_p, const pthread_mutexattr_t* attr_p)`: 初始化互斥锁；第二个参数直接 `NULL`
   2. `int pthread_mutex_lock(pthread_mutex_t* mutex_p)`: 获取临界区访问权限，用互斥锁加锁
   3. `int pthread_mutex_unlock(pthread_mutex_t* mutex_p)`: 离开临界区，使互斥锁解锁
   4. `int pthread_mutex_destroy(pthread_mutex_t* mutex_p)`: 互斥锁销毁
   5. 用互斥量计算全局和
      ```c
      # define PTHREAD_CNT 8
      pthread_mutex_t mutex;
      pthread_mutex_init(&mutex, NULL); 
      double global_sum = 0.0; 

      void* pthread_sum(void* rank) {
         long my_rank = (long)rank; 
         double local_sum = 0.0; 
         double local_n = n / PTHREAD_CNT; 
         int start = my_rank * local_n; 
         int end = start + local_n; 

         for (int i = start; i < end; i++) {
            local_sum += arr[i]; 
         }

         pthread_mutex_lock(&mutex); 
         global_sum += local_cum; 
         pthread_mutex_unlock(&mutex); 

         return NULL; 
      }
      ```
5. 信号量 semaphore 信号量可以认为是一种特殊类型的 `unsigned int` 无符号整型变量。大多数情况下，只给它们赋值0 和1, 这种只有 0 和 1 值的信号量称为二元信号量。粗略地讲，0 对应于上了锁的互斥量，1 对应于未上锁的互斥量。要把一个二元信号量用做互斥量时，需要先把信号量的值初始化为 1, 即开锁状态。在要保护的临界区前调用函数 `sem_wait`, 线程执行到 `sem_wait` 函数时，如果信号量为0, 线程就会被阻塞。如果信号量是非0值，就减 1 后进入临界区。执行完临界区内的操作后，再调用 `send_post` 对信号量的值加 1, 使得在 `sem_wait` 中阻塞的其他线程能够继续运行；我们可以导入`<semaphore.h>` 库来定义信号量 `sem_t` （注意不是 pthreads 库的一部分） 
   1. `int sem_init(sem_t* semaphore_p, int shared, unsigned initial_val)`: 第二个参数为0即可
   2. `int sem_destroy(sem_t* semaphore_p)`
   3. `int sem_post(sem_t* semaphore_p)`
   4. `int sem_wait(sem_t* semaphore_p)`
      ```c
      void* send_msg(void* rank) {
         long my_rank = (long) rank；
         long dest = (my_rank + 1) % THREAD_CNT;
         char* msg = malloc(MSG_MAX * sizeof(char)); 
         sprintf(msg, "Hello to %ld from %ld", dest, my_rank); 
         messages[dest] = msg;

         sem_post(&semaphores[dest]); // unlock the semaphore of dest

         sem_wait(&semaphores[rank]); 
         printf("Thread %ld > %s \n", my_rank, messages[my_rank]); 

         return NULL; 
      }
      ```
6. 路障 Barrier：通过保证所有线程在程序中处于同一个位置来同步线程的同步点；Pthread并没有直接提供barrier的实现，但是可以通过以下三种方法
   1. 忙等待和互斥量：我们使用一个由互斥量保护的共享计数器。当计数器的值表明每个线程都已经进入临界区，所有线程就可以离开忙等待的状态了
   2. 信号量：使用一个计数器来记录进入barrier的线程数量；使用一个 `count_sem` 来保护计数器，另外使用一个 `barrier_sem` 来阻塞已经进入路障的线程
      ```c
      int counter;       // initialize to 0
      sem_t count_sem;   // initialize to 1
      sem_t barrier_sem; // initialize to 0

      void* thread_work(...) {
         ...
         // Barriers 
         sem_wait(&count_sem); 
         if (counter == thread_count - 1) {
            counter = 0; 
            sem_post(&count_sem); 
            for (int i = 0; i < thread_count - 1; i++) {
               sem_post(&barrier_sem);
            }
         }
         else {
            counter++; 
            sem_post(&count_sem);
            sem_wait(&barrier_sem);
         }
      }
      ```
   3. 条件变量：条件变量是一个数据对象，允许线程在某个特定条件或事件发生前都处于挂起状态。当事件或条件发生时，另一个线程可以通过信号来唤醒挂起的线程。一个条件变量总是与一个互斥量相关联；pthreads 中的条件变量类型为 `pthread_cond_t`
      1. `int pthread_cond_init(pthread_cond_t* cond_var_p, const pthread_condattr_t* cond_attr_p)`: 初始化条件变量，第二个参数直接 NULL
      2. `int pthread_cond_destroy(pthread_cond_t* cond_var_p)`: 销毁条件变量
      3. `int pthread_cond_signal(pthread_cond_t* cond_var_p)`: 解锁一个阻塞的线程
      4. `int pthread_cond_broadcast(pthread_cond_t* cond_var_p)`: 解锁所有的阻塞的线程
      5. `int pthread_cond_wait(pthread_cond_t* cond_var_p, pthread_mutex_t* mutex_p)`: 通过互斥量来阻塞线程；相当于按顺序执行了以下操作
         ```c
         pthread_mutex_unlock(&mutex_p); 
         wait_on_signal(&cond_var_p); 
         pthread_mutex_lock(&mutex_p); 
         ```
      6. 用条件变量实现barrier
         ```c
         int counter = 0; 
         pthread_mutex_t mutex; 
         pthread_cond_t cond_var; 
         ...
         void* thread_work(...) {
            ...
            // barrier
            pthread_mutex_lock(&mutex); 
            counter++;
            if (counter == THREAD_CNT) {
               counter = 0; 
               pthread_cond_broadcast(&cond_var); 
            }
            else {
               while (pthread_cond_wait(&cond_varm &mutex) != 0); 
            }
            pthread_mutex_unlock(&mutex); 
            ...
         }
         ```
7. 读写锁：对于链表场景，假设链表节点的构造如下
   ```c
   struct list_node {
      int data; 
      struct list_node* next; 
   }; 
   ```
   1. 如果只是简单的对每次线程要访问链表的时候就加锁，使得其他线程无法访问整个链表，那么基本上就是串行化
      ```c
      pthread_mutex_lock(&list_mutex); 
      member(value); 
      pthread_mutex_unlock(&list_mutex);
      ```
   2. 细粒度化：对于每个节点都给一个锁
      ```c
      struct list_node_s {
         int data; 
         struct list_node_s* next; 
         pthread_mutex_t mutex; 
      }; 
      ```
      然后具体实现如下：每次访问一个结点时，必须先对该结点加锁。注意，这要求有一个与 head_p 指针相关联的互斥量
      ```c
      int Member(int value) {
         struct list_node_s* temp_p; 
         pthread_mutex_lock(&head_p_mutex);
         temp_p = head_p;
         while (temp_p != NULL && temp_p->data < value) {
            if (temp_p->next != NULL)
               pthread_mutex_lock(&(temp_p->next->mutex));
            if (temp_p == head_p)
               pthread_mutex_unlock(&head_p_mutex);
            pthread_mutex_unlock(&(temp_p->mutex)); 
            temp_p = temp_p->next;
         }

         if (temp.p == NULL || temp_p->data > value) {
            if (temp_p == head_p)
               pthread_mutex_unlock(&head_p_mutex);
            if (temp_p != NULL)
               pthread_mutex_unlock(&(temp_p->mutex));
            return 0;
         }
         else {
            if (temp.p == head.p)
               pthread.mutex_unlock(&head_p_mutex);
            pthread_mutex_unlock(&(temp_p->mutex));
            return 1;
         }
      }
      ```
   3. pthreads 读写锁: 提供了 `pthread_rwlock_rdlock` 和 `pthread_rwlock_wrlock` 两个函数，第一个为读操作对读写锁进行加锁，第二个为写操作对读写锁进行加锁。多个线程能通过调用读锁函数而同时获得锁，但只有一个线程能通过写锁函数获得锁。因此，如果任何线程拥有了读锁，则任何请求写锁的线程将阻塞在写锁函数的调用上。而且，如果任何线程拥有了写锁，则任何想获取读或写锁的线程将阻塞在它们对应的锁函数上
      ```c
      pthread_rwlock_rdlock(&rwlock); 
      member(value); 
      pthread_rwlock_unlock(&rwlock);
      ...
      pthread_rwlock_wrlock(&rwlock); 
      insert(value); 
      pthread_rwlock_unlock(&rwlock); 
      ...
      pthread_rwlock_wrlock(&rwlock); 
      delete(value); 
      pthread_rwlock_unlock(&rwlock); 
      ```

## 5. 线程安全性
1. 线程安全：如果一个代码块能够被多个线程同时执行而不引起问题，那么它是线程安全的
2. 伪共享：由于多个线程同时访问不同的变量，但这些变量共享同一缓存行（Cache Line），导致频繁的缓存行无效（Cache Line Invalidation）和更新操作，从而降低程序的性能