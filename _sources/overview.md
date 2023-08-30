# Overview


There are several ways to increase the number of computations per second. Some include:
1. Increasing the number of computations per clock cycle:
    - by adding more transistors to a single CPU
    - by increasing the clock speed of a single CPU
2. Increasing the number of 'workers' crunching computations:
    - by adding more 'cores' to a single machine
    - by adding more machines to a network
    - by using specialized hardware to do the math (e.g., GPUs, TPUs, FPGAs, etc.)


The first approach is called **serial** computing, and the second approach is called **parallel** computing.

Unfortunately, the first approach has hit a wall. See the figures from the [Standford VLSI Group's CPU DB](http://cpudb.stanford.edu/visualize/clock_frequency).
|Features on a chip | Clock speed |
|:------------------|:-----------:|
| ![](static/feature_size.png)         | ![](static/clock_cycle.png)       |

We're unlikely to make drastic improvements in either of these in the near future, due to inherent physical
limitations on the construction of chips and circuit boards.
**Parallel computing is a way forward.**



The pioneer of multiprocessing, and "Father of supercomputing", [Seymour Cray](https://www.britannica.com/biography/Seymour-R-Cray), once said:
> If you were plowing a field, which would you rather use? Two strong oxen or 1024 chickens?”
>
> - Seymour Cray,

![](static/chick_vs_ox.png)

Cray considered this an obvious question, as to him, it is absurd to plough a field with chickens.
He thought it would be absurd to use several processors to crunch
computations in parallel, in place of one fast processor to crunch computations sequentially.
By the early 2000s, views began to shift...





### Flynn's taxonomy and "types" of parallelization





```note
We will focus on the SIMD (Single Instruction Multiple Data) type of parallelization.
```





Very relavent video of [several thousand chickens fighting a few T-rexes](https://www.youtube.com/watch?v=Tc_JWE_ypEk)

