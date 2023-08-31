# Parallel Techniques with Python

The goals of this tutorial:
- Basic concepts of parallelization
- vectorization
- threads/processes
- GPU/TPUs
- High-level python parallelization techniques

We'll go over some [concepts] and [basics] (~10 min + QA) and then break into groups of 2-3 to work on some exercises.

In the exercises, you'll be generating animations like these:
|        [π-estimation]        |         [N-Body sim]          |          [Fractals]           |
|:----------------------------:|:-----------------------------:|:-----------------------------:|
|      Multiprocessing         |       Vectorization           |         GPU-kernel            |
| [![][pi_anim]][π-estimation] | [![][nbody_anim]][N-Body sim] | [![][fractal_anim]][Fractals] |


[pi_anim]: exercises/pi_estimation.gif
[nbody_anim]: exercises/out_nb_np/orbit.gif
[fractal_anim]: https://github.com/avivajpeyi/parallelization_techniques/assets/15642823/87f0494f-c757-4983-921e-99bc7381fc7a

[π-estimation]: exercises/pi_estimator.ipynb
[N-Body sim]: exercises/nbody.ipynb
[Fractals]: exercises/fractal.ipynb

[concepts]: overview.md
[basics]: basics.ipynb