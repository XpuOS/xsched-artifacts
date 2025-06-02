# Get Started

This example shows how to use XSched's to schedule tasks on **NVIDIA GPU**.

## Build XSched

Make sure you have already built XSched with CUDA.

```bash
cd xsched-artifacts/sys/xsched
make cuda
```

## Build the Example

```bash
cd xsched-artifacts/get_started
make # NVCC is required
```

## Run the Example

This example is a simple vector addition program, but running many times.

```bash
./app
```

You will see the output like this:

```
Task 0 completed in 66 ms
Task 1 completed in 66 ms
Task 2 completed in 66 ms
Task 3 completed in 66 ms
Task 4 completed in 66 ms
```

Now, you can open a new terminal and run two apps simultaneously.

```bash
# In the first terminal
./app

# In the second terminal
./app
```

You will see the output like this:

```
# In the first terminal
Task 0 completed in 114 ms
Task 1 completed in 134 ms
Task 2 completed in 143 ms
Task 3 completed in 145 ms
Task 4 completed in 132 ms
```

```
# In the second terminal
Task 13 completed in 78 ms
Task 14 completed in 115 ms
Task 15 completed in 133 ms
Task 16 completed in 144 ms
Task 17 completed in 145 ms
Task 18 completed in 131 ms
```

The two apps have the same performance (double the time as the single app) as the GPU hardware schedules them fairly.

## Run with XSched

Now, let's use XSched to prioritize one of the apps.

First, we should start the XSched server. It is a daemon process for scheduling the GPU processes.
```bash
# Open a new terminal
cd xsched-artifacts/sys/xsched/output/bin
# HPF: High Priority First
# 50000: server listening port, which is used to connect with XCli, our command line tool for XSched
./xserver HPF 50000

# or just use our script
bash ./run_xserver.sh
```

Next, we can run the two apps with XSched.

```bash
# In the first terminal
bash ./run_high_priority.sh

# In the second terminal
bash ./run_low_priority.sh
```

These scripts just modify the environment variables and run the app.

You will see the output like this:

```
# In the first terminal
Task 25 completed in 67 ms
Task 26 completed in 67 ms
Task 27 completed in 67 ms
Task 28 completed in 80 ms
Task 29 completed in 78 ms
Task 30 completed in 69 ms
```

```
# In the second terminal
Task 1 completed in 207 ms
Task 2 completed in 195 ms
Task 3 completed in 165 ms
Task 4 completed in 184 ms
Task 5 completed in 188 ms
Task 6 completed in 189 ms
Task 7 completed in 172 ms
```

The high priority app achieves a similar performance as the standalone execution, while the low priority app is slowed down.

You can also use `xcli` to check the status of the XQueues and manage them.

```bash
# Open a new terminal
cd xsched-artifacts/sys/xsched/output/bin

# show help
./xcli -h

# show all XQueues, ctrl-c to stop
# -f: refresh frequency
./xcli top -f 10

# give a hint to set the priority of XQueue 0xaf246296bbdf3260 to 2
./xcli hint -x 0xaf246296bbdf3260 -p 2
```
