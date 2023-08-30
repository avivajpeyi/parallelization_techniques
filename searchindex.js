Search.setIndex({"docnames": ["basics", "exercises/fractal", "exercises/nbody", "exercises/pi_estimator", "index", "overview", "sources/7-gpu", "sources/fractal_1027", "sources/mpi4py", "sources/parallel_overview"], "filenames": ["basics.ipynb", "exercises/fractal.ipynb", "exercises/nbody.ipynb", "exercises/pi_estimator.ipynb", "index.md", "overview.md", "sources/7-gpu.ipynb", "sources/fractal_1027.ipynb", "sources/mpi4py.ipynb", "sources/parallel_overview.ipynb"], "titles": ["Py Parallelization Basics", "Fractal Generation", "N-Body Simulation", "\u03c0-Estimator", "Parallel Techniques with Python", "Overview", "GPU: CuPy, Numba-GPU, PyCUDA", "Fun With Fractals and CuPy on GPU", "&lt;no title&gt;", "Parallelization"], "terms": {"load_ext": [0, 7], "autoreload": [0, 7], "2": [0, 1, 2, 3, 4, 6, 8, 9], "start": [0, 2, 3, 7, 9], "simpl": [0, 3], "exampl": [0, 1, 3, 5, 9], "import": [0, 1, 2, 3, 5, 6, 7, 8, 9], "numpi": [0, 3, 6], "np": [0, 1, 2, 3, 5, 7, 9], "from": [0, 1, 2, 3, 5, 6, 8, 9], "time": [0, 1, 2, 3, 5, 6, 9], "process_tim": [0, 1, 3], "matplotlib": [0, 1, 2, 3, 6, 7, 9], "pyplot": [0, 1, 2, 3, 6, 7, 9], "plt": [0, 1, 2, 3, 6, 7, 9], "tqdm": [0, 1, 2, 3], "auto": [0, 1, 2, 3, 8], "warn": [0, 2], "def": [0, 1, 2, 3, 6, 7, 9], "fn": 0, "x": [0, 1, 2, 3, 5, 6, 7, 8, 9], "return": [0, 1, 2, 3, 6, 7, 9], "runtime_loop": 0, "t0": [0, 1, 3], "row": 0, "col": 0, "shape": [0, 1, 2, 6, 7], "i": [0, 1, 2, 3, 5, 6, 7, 8, 9], "rang": [0, 1, 2, 3, 5, 6, 7, 9], "j": [0, 2, 7], "runtime_np_vector": 0, "n": [0, 1, 3, 4, 5, 6, 8, 9], "1000": [0, 1, 5, 7, 9], "random": [0, 2, 3, 5, 7, 9], "randn": [0, 2, 5, 9], "astyp": 0, "dtype": [0, 1, 2, 3, 6, 7], "float32": [0, 1, 7], "print": [0, 1, 2, 3, 6, 7, 8], "f": [0, 1, 2, 3, 7, 8, 9], "loop": [0, 1, 3, 6, 7, 9], "3f": [0, 3, 7], "s": [0, 1, 2, 3, 6, 7, 8, 9], "For": [0, 1, 5, 6, 7, 9], "thing": [0, 1, 5, 7], "work": [0, 2, 3, 4, 5, 6, 7, 9], "better": [0, 1, 3, 6], "you": [0, 1, 2, 3, 5, 6, 7, 9], "first": [0, 3, 5, 6, 7], "save": [0, 1, 2, 3, 8], "code": [0, 1, 2, 3, 5, 6, 9], "file": [0, 3, 8], "function": [0, 1, 2, 3, 6, 7, 9], "thi": [0, 1, 2, 3, 4, 5, 6, 7, 9], "becaus": [0, 5, 6, 9], "modul": [0, 7], "need": [0, 3, 5, 6, 7, 9], "new": [0, 7, 9], "process": [0, 3, 4, 5, 6, 8, 9], "abl": [0, 5, 7], "find": [0, 9], "thei": [0, 5, 6, 7], "ar": [0, 1, 2, 3, 5, 6, 7, 8, 9], "defin": [0, 1], "notebook": 0, "cell": [0, 7], "writefil": [0, 3], "basics_multi_demo": 0, "mp": [0, 3], "n_cpu": 0, "cpu_count": [0, 3], "runtime_multiprocess": 0, "pool": [0, 3, 9], "_": [0, 2, 3, 7], "map": [0, 3], "runtime_multithread": 0, "threadpool": 0, "why": [0, 1, 2, 3, 6, 9], "slower": [0, 1], "There": [0, 2, 5, 6], "overhead": [0, 7, 9], "associ": [0, 9], "manag": [0, 5, 9], "thread": [0, 4, 5, 6, 9], "so": [0, 3, 7, 8, 9], "don": [0, 1, 3, 5, 6, 7, 9], "t": [0, 1, 2, 3, 5, 6, 7, 8, 9], "want": [0, 5, 7], "us": [0, 1, 2, 3, 5, 6, 8, 9], "task": [0, 5, 9], "like": [0, 1, 5, 6, 7, 9], "python": [0, 1, 3, 5, 6, 7, 9], "great": [0, 7, 9], "cpu": [0, 5, 6, 9], "comput": [0, 1, 3, 6, 7, 9], "perfect": [0, 6], "o": [0, 2, 8], "oper": [0, 5, 6, 7], "web": [0, 5], "scrape": 0, "processor": [0, 5, 8, 9], "sit": 0, "idl": 0, "wait": [0, 6], "data": [0, 1, 2, 3, 5, 6, 9], "intens": [0, 9], "do": [0, 1, 3, 5, 7, 9], "math": [0, 3, 5, 6], "littl": 0, "benefit": 0, "best": [0, 9], "suit": 0, "bound": [0, 1], "while": [0, 5, 9], "test": [0, 2, 9], "two": [0, 2, 5, 7, 9], "graphic": [0, 5], "unit": [0, 5, 6], "librari": [0, 5, 6, 9], "cupi": [0, 2, 3], "jax": [0, 1, 3], "tensor": 0, "well": [0, 5, 7, 9], "On": [0, 6, 9], "colab": [0, 7, 8], "mai": [0, 6], "chang": [0, 1, 6, 9], "your": [0, 1, 2, 3, 5, 9], "runntim": 0, "type": [0, 1, 2, 7, 8], "hardwar": [0, 5, 9], "acceler": [0, 2, 7, 9], "try": [0, 1, 2, 3, 6, 7, 8, 9], "cp": [0, 1, 7], "check": [0, 3], "current": [0, 6], "devic": [0, 6, 7], "platform": 0, "cuda": [0, 1, 6], "cupy_instal": 0, "true": [0, 2, 3, 6, 7, 9], "except": [0, 2, 6, 8, 9], "importerror": [0, 8], "fals": [0, 2, 3, 6, 7, 9], "instal": [0, 6, 8], "jnp": 0, "jit": [0, 6, 9], "lib": [0, 6, 8], "xla_bridg": 0, "get_backend": 0, "jax_instal": 0, "jax_fn": 0, "runtime_jax": 0, "nan": [0, 1, 2], "arrai": [0, 1, 2, 3, 6, 7, 9], "block_until_readi": 0, "runtime_cupi": 0, "get": [0, 1, 2, 6, 7, 8, 9], "runtime_func": 0, "dict": [0, 1, 2, 3], "np_vector": 0, "collect_runtime_data": 0, "n_val": [0, 2, 3], "n_trial": [0, 2, 3], "5": [0, 1, 2, 3, 6, 8, 9], "k": [0, 2, 3], "kei": [0, 2], "enumer": [0, 1, 2, 3, 7], "total": [0, 2, 8, 9], "len": [0, 2, 3], "item": [0, 3], "trial": [0, 2, 3], "empti": 0, "1": [0, 1, 2, 3, 6, 7, 8], "argsort": 0, "append": [0, 1], "quantil": [0, 2], "0": [0, 1, 2, 3, 6, 7, 8, 9], "05": [0, 1, 2], "95": [0, 1, 2], "plot_runtim": [0, 1, 2], "fig": [0, 1, 2, 3, 6, 7], "ax": [0, 1, 2, 3, 6, 7], "subplot": [0, 1, 2, 3, 6, 7], "figsiz": [0, 1, 2, 3, 6, 7], "10": [0, 1, 2, 3, 4, 7, 8, 9], "6": [0, 1, 3, 5, 7, 8], "v": [0, 2, 8, 9], "label": [0, 1, 2, 3], "color": [0, 1, 2, 3], "c": [0, 1, 2, 3, 6, 8], "fill_between": [0, 2, 3], "alpha": [0, 1, 2, 3], "set_xlabel": [0, 1, 2, 3, 7], "size": [0, 1, 2, 3, 6, 7, 8, 9], "set_ylabel": [0, 1, 2, 3, 7], "set_yscal": [0, 1, 2], "log": [0, 1, 2, 3, 7], "set_xscal": [0, 1, 3], "set_xlim": [0, 2, 3], "min": [0, 2, 3, 4, 7], "max": [0, 2, 3, 7, 9], "legend": [0, 1, 2, 3], "fontsiz": [0, 1, 2, 3], "15": [0, 6], "frameon": [0, 2, 3], "geomspac": [0, 2, 3], "1e2": 0, "1e3": 0, "int": [0, 1, 2, 3, 7, 8], "accord": 1, "wikipedia": [1, 7], "A": [1, 5, 6, 7], "geometr": [1, 7], "contain": 1, "detail": [1, 6], "structur": [1, 7], "arbitrarili": 1, "small": [1, 9], "scale": [1, 5, 6], "julia": [1, 7], "set": [1, 2, 6, 7, 8], "we": [1, 2, 3, 4, 5, 6, 7, 9], "ll": [1, 2, 3, 4, 7], "focu": [1, 5, 9], "made": 1, "complex": [1, 2, 3, 7], "number": [1, 2, 5, 7, 8, 9], "One": 1, "all": [1, 2, 3, 5, 6, 7, 9], "z": [1, 2, 6, 8], "given": [1, 2, 7], "bi": [1, 7], "sequenc": 1, "z_": 1, "z_n": 1, "remain": 1, "fuction": 1, "updat": [1, 2, 3, 7, 8], "iter": [1, 3, 7], "initi": [1, 8], "variabl": 1, "valu": [1, 2, 3, 6, 7], "base": [1, 7], "upon": 1, "necessari": 1, "determin": 1, "whether": 1, "unbound": [1, 7], "often": [1, 5, 9], "threshold": 1, "prevent": 1, "infinit": [1, 7], "which": [1, 2, 5, 7, 8, 9], "can": [1, 2, 3, 5, 6, 7, 9], "one": [1, 5, 6, 7, 9], "both": [1, 7, 9], "surpass": 1, "below": [1, 2, 3], "stop": [1, 2], "when": [1, 6, 7, 9], "4": [1, 2, 3, 6, 8, 9], "predefin": 1, "either": [1, 9], "method": [1, 2, 3, 6, 7], "trend": 1, "toward": 1, "infin": 1, "visualis": 1, "To": 1, "visual": [1, 7], "locat": [1, 8], "pixel": [1, 7], "dimension": [1, 6], "imag": [1, 6, 7], "real": [1, 6, 7], "portion": 1, "index": [1, 7], "imaginari": [1, 7], "y": [1, 2, 3, 5, 6, 8, 9], "vice": 1, "versa": 1, "each": [1, 2, 5, 6, 7, 9], "its": [1, 5, 6, 7, 9], "plug": 1, "result": [1, 3, 7], "x1": 1, "x2": 1, "width": [1, 2, 3, 6], "y1": 1, "y2": 1, "height": [1, 6], "final": [1, 3, 8], "could": [1, 7], "black": 1, "colormap": 1, "displai": 1, "normal": [1, 7], "e": [1, 5, 6, 7, 8, 9], "g": [1, 2, 5, 6, 8], "255": 1, "8": [1, 2, 6, 7, 8, 9], "bit": [1, 9], "depth": 1, "The": [1, 3, 4, 5, 6, 9], "fractan": 1, "constant": [1, 2, 7, 9], "maximum": 1, "allow": [1, 8], "resolut": 1, "grid": [1, 6, 7, 9], "extent": 1, "versu": 1, "an": [1, 2, 3, 5, 7, 9], "interact": 1, "plotter": 1, "found": [1, 8], "here": [1, 2, 3, 5, 7, 9], "768": [1, 7], "epsilon": 1, "1e": [1, 2, 7], "mesh_min": 1, "mesh_max": 1, "mesh_siz": 1, "mesh_r": [1, 7], "mesh_im": [1, 7], "meshgrid": [1, 7, 9], "arang": [1, 7], "compute_cpu_julia_grid": 1, "np_zmesh_r": [1, 7], "np_zmesh_im": [1, 7], "constant_r": 1, "constant_imag": 1, "pre": [1, 7], "mesh": [1, 7], "param": [1, 7], "part": [1, 2, 7], "repres": [1, 7], "nr": [1, 7], "nc": [1, 7], "max_escape_it": [1, 7], "fractal_imag": 1, "zero": [1, 2, 3, 6, 7], "r": [1, 2, 3, 7, 8], "b": [1, 7, 8], "temp_real": [1, 7], "temp_imag": [1, 7], "break": [1, 2, 4, 6, 7, 9], "go": [1, 4, 7, 9], "diverg": [1, 6], "els": [1, 3, 5, 6, 7, 8], "log2": [1, 7], "float": [1, 2, 5, 7, 9], "156": [1, 7], "julia_fract": 1, "2f": [1, 3, 7], "plot_julia_grid": 1, "none": [1, 7], "magma": 1, "imshow": [1, 6, 7], "cmap": [1, 2, 3], "text": [1, 2, 3, 7, 8], "top": [1, 2, 3, 9], "left": [1, 2, 3, 7], "corner": [1, 2], "transform": [1, 2, 3, 7], "transax": [1, 2, 3], "ha": [1, 3, 5, 6, 7, 9], "va": [1, 3], "white": [1, 2, 3], "bbox": [1, 3], "boxstyl": [1, 3], "round": [1, 3], "facecolor": [1, 3], "9": [1, 2, 3, 7], "axi": [1, 2, 3, 7], "off": [1, 7, 9], "remov": [1, 2, 3, 7, 8], "space": [1, 5, 9], "around": [1, 2, 6], "subplots_adjust": [1, 3], "right": [1, 2, 3, 7], "bottom": 1, "although": [1, 7, 9], "interfac": 1, "write": [1, 6, 7, 8], "actual": [1, 2, 7, 9], "kernel": [1, 4], "specif": [1, 7], "more": [1, 2, 3, 5, 6, 7, 9], "technic": [1, 7], "also": [1, 3, 6], "flexibl": [1, 9], "access": [1, 9], "share": [1, 2, 3, 5, 8, 9], "memori": [1, 5, 6, 7, 9], "etc": [1, 5], "see": [1, 5, 6, 7, 9], "doc": 1, "creat": [1, 9], "instead": [1, 6, 7, 9], "gpu_mesh_r": 1, "gpu_mesh_im": 1, "follow": [1, 3, 5, 7, 8], "compli": 1, "upload": [1, 7], "compute_gpu_julia_grid": 1, "elementwisekernel": 1, "complex_grid_r": 1, "complex_grid_im": 1, "out": [1, 3, 7, 8, 9], "zn_real": 1, "zn_imag": 1, "log2f": [1, 7], "0f": [1, 7], "gpu_znplusc": [1, 7], "sometim": 1, "won": 1, "have": [1, 2, 5, 6, 7, 9], "incur": 1, "ani": [1, 2, 3, 7], "cost": 1, "cde": 1, "compil": [1, 5, 6, 7, 9], "dure": [1, 9], "execut": [1, 5, 9], "deal": 1, "transfer": [1, 7], "onli": [1, 7], "end": 1, "c_val": [1, 7], "7885": 1, "335": 1, "re": [1, 2, 5, 6, 7], "im": [1, 7], "collect_cpu_runtim": 1, "resolution_s": 1, "desc": [1, 2, 3], "collect_gpu_runtim": 1, "cpu_runtim": 1, "gpu_runtim": 1, "32": [1, 2, 6, 7], "64": 1, "128": [1, 7], "256": [1, 2, 7], "512": 1, "14": [1, 2, 3], "1024": [1, 5, 7], "savefig": [1, 2, 3], "runtimes_fract": 1, "png": [1, 2, 3, 7], "make": [1, 2, 3, 5, 6, 7, 9], "ib": 1, "co": [1, 3, 9], "theta": [1, 3], "sin": [1, 3], "pi": [1, 3, 9], "some": [1, 2, 3, 4, 5, 6, 7, 9], "select": [1, 2, 7, 9], "zoom": [1, 7], "imageio": [1, 2, 3], "make_julia_gif": 1, "c_mag": 1, "outnam": [1, 2, 3], "gif": [1, 2, 3, 7], "theta_v": 1, "linspac": [1, 2, 3, 9], "100": [1, 3, 7, 8], "get_writ": [1, 3], "mode": [1, 3, 6, 8], "writer": [1, 3], "temp": [1, 3], "dpi": [1, 2, 3], "append_data": [1, 3], "imread_v2": [1, 2, 3], "close": [1, 2, 3, 7], "julia_7885": 1, "julia_335": 1, "what": [1, 2, 3, 7, 9], "faster": [1, 7, 9], "than": [1, 6, 7, 9], "increas": [1, 2, 3, 5, 9], "happen": [1, 2, 3, 5, 9], "differ": [1, 2, 3, 5, 6, 7, 9], "between": [1, 7, 9], "Is": 1, "speed": [1, 2, 3, 5, 9], "up": [1, 2, 3, 5, 7, 9], "releas": [1, 2, 3, 8], "answer": [1, 2, 3, 6], "after": [1, 2, 3, 7], "workshop": [1, 2, 3], "pleas": [1, 2, 3, 7], "remind": [1, 2, 3], "me": [1, 2, 3], "lol": [1, 2, 3], "cool": [1, 2, 3], "down": [1, 2, 3, 7, 9], "websit": [1, 2, 3], "version": [1, 2, 3, 7, 8, 9], "page": [1, 2, 3], "baselin": 2, "compar": [2, 3, 7], "against": 2, "multiprocess": [2, 4], "main": [2, 8, 9], "newtonian_acceler": 2, "po": 2, "mass": 2, "soften": 2, "calcul": [2, 5, 9], "particl": 2, "due": [2, 5, 6, 7, 9], "gravit": 2, "forc": 2, "other": [2, 3, 6, 7, 9], "most": [2, 5, 7, 9], "computation": 2, "expens": 2, "nbody_runn": 2, "run": [2, 5, 6, 8, 9], "step": [2, 7, 9], "util": [2, 5, 6], "os": 2, "glob": 2, "datetim": 2, "trang": 2, "rcparam": [2, 3], "colorsi": 2, "colorconvert": 2, "linearsegmentedcolormap": 2, "xtick": [2, 3], "major": [2, 3, 9], "pad": [2, 3], "7": [2, 3, 6, 7, 8, 9], "minor": [2, 3], "3": [2, 3, 4, 6, 8, 9], "ytick": [2, 3], "font": [2, 3], "20": [2, 3, 5, 6, 8, 9], "direct": [2, 3], "collect_runtim": 2, "func": [2, 3], "kwarg": 2, "ndarrai": 2, "collect": [2, 9], "input": [2, 7], "paramet": 2, "list": [2, 8], "option": 2, "default": [2, 6], "keyword": [2, 6], "argument": [2, 6, 7], "pass": [2, 7], "filterwarn": 2, "error": [2, 6], "npart_i": 2, "trial_i": 2, "now": [2, 9], "get_runtim": 2, "runtimewarn": 2, "total_second": 2, "make_gif": 2, "im_regex": 2, "durat": 2, "img": 2, "sort": 2, "lambda": 2, "findal": 2, "d": [2, 3, 6, 7, 8, 9], "frame": [2, 7], "mimsav": 2, "remove_spin": [2, 3], "spine": [2, 3], "tick": [2, 3], "set_vis": [2, 3], "set_ytick": [2, 3, 7], "set_xtick": [2, 3, 7], "scale_color_bright": 2, "scale_l": 2, "rgb": 2, "to_rgb": 2, "convert": 2, "hl": 2, "h": [2, 6, 8], "l": [2, 8], "rgb_to_hl": 2, "manipul": 2, "hls_to_rgb": 2, "make_colormap": 2, "30": [2, 3, 7, 8], "from_list": 2, "custom_": 2, "revers": 2, "union": 2, "outdir": 2, "orbit_out": 2, "newtonian_acceleration_bas": 2, "posit": 2, "p": [2, 8], "matrix": [2, 7], "store": 2, "pairwis": 2, "separ": 2, "r_j": 2, "r_i": 2, "dx": 2, "dy": 2, "dz": 2, "inv_r3": 2, "sum": [2, 3, 7], "ay": 2, "az": 2, "pack": 2, "togeth": [2, 7], "compon": [2, 7, 8], "nbody_runner_bas": 2, "tend": 2, "dt": 2, "01": [2, 3], "random_se": 2, "17": 2, "max_runtim": 2, "verbos": 2, "seed": [2, 3], "initialis": 2, "randomli": [2, 9], "veloc": 2, "vel": 2, "nt": 2, "ceil": [2, 6], "runtime_start": 2, "vel_mean": 2, "m": [2, 5, 6, 9], "zip": 2, "acc": 2, "pos_sav": 2, "disabl": 2, "exceed": 2, "second": [2, 5, 7, 9], "transpos": 2, "timeit": [2, 9], "ms": [2, 9], "ns": 2, "per": [2, 5, 7], "mean": [2, 3, 5, 9], "std": [2, 3, 8], "dev": [2, 8], "let": [2, 3, 7, 9], "plot": 2, "plot_particl": 2, "n_time_tot": 2, "tab": [2, 3], "blue": [2, 3], "2d": [2, 7], "should": [2, 7, 9], "n_particl": 2, "xyz": 2, "n_time": 2, "trail": 2, "str": [2, 7], "n_part": 2, "figur": [2, 5, 7], "80": 2, "gca": 2, "orbit": 2, "idx_end": 2, "argmax": [2, 7], "where": [2, 9], "idx_start": 2, "nidx": 2, "max_siz": 2, "scatter": [2, 3], "ec": 2, "lw": [2, 3], "mask": [2, 7], "zorder": [2, 3], "set_ylim": 2, "set_aspect": [2, 3], "equal": [2, 3], "box": [2, 7], "border": 2, "tight_layout": [2, 3], "plot_particle_gif": 2, "dur": 2, "makedir": 2, "exist_ok": 2, "add": [2, 3, 7], "textbox": 2, "003d": 2, "verticalalign": 2, "fontstyl": 2, "ital": 2, "orbit_": 2, "out_bas": 2, "swap": 2, "statement": [2, 7], "ve": [2, 3, 5, 9], "unvector": 2, "optim": [2, 6, 7], "yourself": 2, "newtonian_acceleration_np": 2, "hstack": 2, "matmul": 2, "nbody_runner_np": 2, "bool": 2, "ones": 2, "68": 2, "13": [2, 7], "81": 2, "repeat": [2, 7], "averag": 2, "40": [2, 6, 8], "nbody_runtim": 2, "slightli": [2, 7], "longer": 2, "out_nb_np": 2, "about": [2, 3], "think": [2, 3], "wai": [2, 3, 5, 7, 9], "gpu": [2, 3, 4], "feel": 2, "free": 2, "past": 2, "chatgpt": 2, "ask": [2, 6], "help": 2, "bonu": 2, "dont": 2, "mont": [3, 9], "carlo": [3, 9], "varieti": 3, "techniqu": [3, 5, 6], "involv": [3, 5], "repeatedli": [3, 7], "sampl": [3, 7], "obtain": [3, 9], "numer": [3, 9], "outcom": 3, "easi": [3, 7], "algorithm": [3, 8], "approxim": 3, "imagin": 3, "circl": [3, 9], "radiu": [3, 9], "center": 3, "inscrib": 3, "squar": [3, 7], "plane": [3, 7], "side": 3, "2r": 3, "point": [3, 5, 7, 9], "within": 3, "how": [3, 7, 9], "mani": [3, 5, 6, 9], "fall": [3, 6], "x\u00b2": 3, "y\u00b2": 3, "By": [3, 5, 9], "count": [3, 9], "improv": [3, 5], "accuraci": 3, "area": 3, "4r\u00b2": 3, "\u03c0r\u00b2": 3, "frac": 3, "rm": [3, 8], "impli": 3, "later": [3, 7, 9], "pi_estim": 3, "py": [3, 6, 8], "100_000": 3, "circle_point": 3, "square_point": 3, "rand_x": 3, "uniform": [3, 7, 9], "rand_i": 3, "origin_dist": 3, "overwrit": 3, "pi_estimation_with_unc": 3, "pi_val": 3, "10_000": 3, "mc": 3, "000": [3, 7], "135": 3, "012": 3, "050": 3, "num_cpu": 3, "parallel_pi_estim": 3, "num_sampl": 3, "num_process": 3, "unrwap": 3, "call": [3, 5, 7, 9], "join": 3, "12": [3, 5, 6, 8], "140": [3, 8], "017": 3, "008": 3, "panda": 3, "pd": 3, "compute_runtime_and_valu": 3, "pi_unc": 3, "datafram": 3, "plot_runtimes_and_pi_valu": 3, "sharex": 3, "axhlin": 3, "ls": 3, "hspace": 3, "parallel": [3, 6], "serial": [3, 5], "pi_estimation_runtim": 3, "bbox_inch": 3, "tight": [3, 9], "much": [3, 5, 7], "further": [3, 7, 9], "fun": 3, "plot_pi_estim": 3, "rand_xi": 3, "coolwarm": 3, "red": 3, "ab": [3, 6, 7], "green": 3, "make_pi_estimation_gif": 3, "max_n": 3, "1000_000": 3, "n_frame": 3, "gener": [3, 9], "multiprocessig": 3, "basic": [3, 4], "simul": [3, 7, 9], "vector": [3, 4, 6, 7, 9], "And": [3, 6, 7], "goal": [4, 7], "tutori": [4, 5], "concept": 4, "tpu": [4, 5], "high": [4, 7], "level": [4, 9], "over": [4, 7, 9], "qa": 4, "group": [4, 5], "exercis": 4, "\u03c0": 4, "estim": 4, "bodi": 4, "sim": 4, "fractal": [4, 6], "sever": [5, 9], "includ": [5, 8], "clock": [5, 9], "cycl": 5, "ad": [5, 9], "transistor": 5, "singl": [5, 6, 7, 9], "worker": 5, "crunch": 5, "core": [5, 6, 9], "machin": [5, 9], "network": [5, 9], "special": [5, 6, 9], "fpga": 5, "approach": [5, 9], "unfortun": 5, "hit": [5, 7], "wall": [5, 9], "standford": 5, "vlsi": 5, "db": 5, "featur": [5, 6, 9], "chip": [5, 9], "physic": [5, 9], "limit": [5, 6, 9], "construct": [5, 7, 9], "circuit": [5, 9], "board": [5, 9], "unlik": [5, 9], "drastic": 5, "perform": [5, 7], "forward": 5, "sisd": 5, "instruct": [5, 9], "stream": [5, 8], "good": 5, "old": 5, "sequenti": 5, "simd": [5, 6], "same": [5, 6, 7, 9], "multipl": [5, 7, 9], "embarrassingli": 5, "misd": 5, "sai": 5, "standard": 5, "deviat": 5, "dataset": 5, "anoth": [5, 6, 7], "mimd": 5, "laptop": 5, "It": [5, 7, 9], "eg": 5, "browser": 5, "word": [5, 9], "script": 5, "vertic": [5, 6, 7], "effect": [5, 6, 7, 9], "md": 5, "If": [5, 6, 7, 9], "fulli": [5, 6], "okai": [5, 6, 7], "someon": [5, 6], "fill": [5, 6], "gap": [5, 6], "them": [5, 6, 7, 9], "concurr": [5, 9], "chain": [5, 9], "program": [5, 9], "across": [5, 9], "refer": [5, 6, 7], "cluster": [5, 9], "own": [5, 6, 9], "similar": [5, 7, 9], "nativ": [5, 7, 9], "support": [5, 6], "legaci": [5, 9], "design": [5, 6, 9], "scientif": [5, 9], "numba": 5, "bypass": 5, "restrict": [5, 9], "effici": [5, 6, 7, 9], "been": [5, 9], "especi": 5, "cleverli": [5, 9], "emploi": 5, "illustr": 5, "htop": [5, 9], "termin": 5, "session": [5, 6], "linalg": [5, 9], "eigval": [5, 9], "output": [5, 7, 9], "system": [5, 7, 9], "monitor": [5, 9], "my": [5, 6, 7], "Not": [5, 7], "were": [5, 7], "full": [5, 6, 9], "routin": [5, 9], "neatli": [5, 9], "split": [5, 7, 9], "distribut": [5, 9], "pioneer": 5, "father": 5, "supercomput": 5, "seymour": 5, "crai": 5, "onc": [5, 7, 9], "said": 5, "plow": 5, "field": 5, "would": [5, 6, 7], "rather": [5, 7], "strong": 5, "oxen": 5, "chicken": 5, "consid": 5, "obviou": 5, "question": [5, 6], "him": 5, "absurd": [5, 7], "plough": 5, "he": 5, "thought": 5, "place": 5, "fast": [5, 7, 9], "earli": 5, "2000": [5, 6], "view": 5, "began": 5, "shift": [5, 7], "flop": 5, "larg": [5, 7, 9], "wont": 5, "replac": [5, 6], "few": [5, 7, 9], "veri": [5, 7, 9], "In": [5, 7, 9], "heavi": 5, "lift": 5, "flow": 5, "relav": 5, "video": 5, "thousand": [5, 7], "fight": 5, "rex": 5, "conda": 6, "cudatoolkit": 6, "export": 6, "cflag": 6, "fpermiss": 6, "pip": 6, "cach": 6, "dir": 6, "luck": 6, "befor": 6, "rowwis": 6, "ever": 6, "columnar": 6, "visibl": 6, "analyst": 6, "33mwarn": 6, "skip": [6, 8], "usr": [6, 8], "local": [6, 8, 9], "python3": [6, 8], "11": [6, 8], "site": 6, "packag": [6, 7], "25": [6, 7], "py3": 6, "egg": 6, "info": 6, "invalid": 6, "metadata": 6, "entri": 6, "name": [6, 7], "0m": 6, "33m": 6, "0mrequir": 6, "alreadi": [6, 7, 8, 9], "satisfi": 6, "57": [6, 8], "requir": [6, 7, 9], "llvmlite": 6, "41": [6, 8], "0dev0": 6, "21": 6, "24": [6, 7], "run_numba_loop": 6, "maxiter": 6, "w": [6, 7], "var": 6, "folder": 6, "qt": 6, "rxjvm_j566v9qn7g754s1v9hzb3p7f": 6, "ipykernel_28676": 6, "2476333802": 6, "numbadeprecationwarn": 6, "1mthe": 6, "nopython": [6, 9], "wa": [6, 7], "suppli": [6, 7], "decor": 6, "implicit": 6, "59": 6, "http": [6, 8], "readthedoc": 6, "io": [6, 8], "en": 6, "stabl": 6, "deprec": 6, "html": 6, "object": [6, 8], "back": [6, 7], "behaviour": 6, "doe": [6, 7], "element": 6, "stage": 6, "prepar": 6, "ogrid": 6, "1j": 6, "int32": [6, 7], "without": [6, 7], "divnow": 6, "alwai": 6, "appli": [6, 7, 9], "am": 6, "nstruction": 6, "ultipl": 6, "ata": 6, "librrari": 6, "implement": [6, 9], "4000": 6, "6000": 6, "starttim": 6, "exactli": 6, "sec": [6, 7, 8], "wonder": 6, "reduc": [6, 7], "problem": [6, 7, 9], "smaller": 6, "previou": [6, 7], "couldn": 6, "fit": [6, 7], "catch": [6, 7], "adher": 6, "api": 6, "isn": [6, 9], "had": [6, 7], "absolut": [6, 7], "wasn": 6, "valueerror": 6, "err": 6, "nevertheless": [6, 9], "expect": [6, 7, 9], "becom": 6, "complet": 6, "peopl": 6, "report": 6, "miss": 6, "as_cuda": 6, "leav": [6, 7], "still": [6, 7], "run_numba": 6, "device_arrai": 6, "doesn": [6, 7], "suffer": 6, "issu": 6, "intermedi": [6, 9], "copi": [6, 7], "8000": 6, "12000": 6, "That": 6, "take": [6, 7, 9], "half": [6, 7], "minut": 6, "project": 6, "pure": 6, "saniti": 6, "sake": 6, "verifi": 6, "inde": 6, "draw": [6, 7, 9], "our": [6, 7, 9], "inlin": [6, 7, 9], "3000": 6, "axesimag": 6, "0x11ff37990": 6, "nich": 6, "directli": [6, 7], "subset": 6, "blog": 7, "post": 7, "tbd": 7, "demonstr": 7, "power": 7, "clone": 7, "cleanli": 7, "rapid": 7, "ecosystem": 7, "even": 7, "though": 7, "note": 7, "googl": [7, 8], "widget": 7, "jupyt": 7, "ran": 7, "linux": [7, 8, 9], "jupyterlab": 7, "mpl_interact": 7, "txt": 7, "howev": [7, 9], "might": [7, 9], "begin": 7, "mathemat": 7, "produc": 7, "surprisingli": 7, "unintut": 7, "pattern": 7, "recurs": 7, "seen": 7, "anim": 7, "show": [7, 9], "mandelbrot": 7, "relat": 7, "suffici": 7, "wide": 7, "datatyp": 7, "custom": 7, "speedup": [7, 8], "enough": 7, "combin": 7, "abov": [7, 9], "th": 7, "nvidia": 7, "geforc": 7, "rtx": 7, "3090": 7, "amd": 7, "ryzen": 7, "5800x": 7, "dumb": 7, "naiv": 7, "probabl": [7, 9], "never": 7, "123": 7, "fp": 7, "xfer": 7, "171": 7, "stai": 7, "389": 7, "spectrogram": 7, "interest": 7, "worth": 7, "know": 7, "pretti": 7, "768x768": 7, "00974": 7, "103": 7, "00233": 7, "429": 7, "00020": 7, "00097": 7, "036": 7, "pil": 7, "__future__": 7, "print_funct": 7, "ipywidget": 7, "fix": 7, "interact_manu": 7, "ipyplot": 7, "iplt": 7, "modulenotfounderror": 7, "traceback": 7, "recent": [7, 9], "last": [7, 9], "line": 7, "get_ipython": 7, "run_line_mag": 7, "No": 7, "super": 7, "timer": 7, "class": 7, "simple_tim": 7, "simpletim": 7, "sure": 7, "properli": 7, "quick": 7, "lulz": 7, "test_siz": 7, "10000": 7, "cupy_mtrx": 7, "reshap": 7, "4096x4096": 7, "addit": [7, 9], "ones_lik": 7, "n_iter": 7, "mtrx": 7, "multipli": 7, "anywai": 7, "less": [7, 9], "npa": 7, "npb": 7, "npc": 7, "gpu_spe": 7, "get_iter_per_sec": 7, "cpu_spe": 7, "speed_ratio": 7, "matric": [7, 9], "huge": [7, 9], "impress": 7, "simplest": 7, "possibl": [7, 9], "look": [7, 9], "escap": 7, "just": [7, 9], "cram": 7, "equival": 7, "op": 7, "individu": [7, 9], "highli": [7, 9], "paralleliz": 7, "clearli": 7, "sizabl": 7, "wouldn": 7, "thu": 7, "give": 7, "mild": 7, "ep": 7, "manual": [7, 8], "outsid": 7, "realloc": 7, "recomput": 7, "cpu_zn2plusc": 7, "c_real": 7, "c_imag": 7, "buffer": 7, "magnitud": 7, "come": 7, "zmesh_real": 7, "zmesh_imag": 7, "156i": 7, "known": 7, "helper": 7, "rescal": 7, "case": [7, 9], "relabel_ax": 7, "rmin": 7, "rmax": 7, "ni": 7, "imin": 7, "imax": 7, "r_width": 7, "new_rtick": 7, "tolist": 7, "new_rticklabel": 7, "1f": 7, "set_xticklabel": 7, "i_width": 7, "new_itick": 7, "new_iticklabel": 7, "set_yticklabel": 7, "cpu_fract": 7, "set_titl": 7, "nameerror": 7, "get_sec_per_it": 7, "notic": 7, "switch": [7, 9], "nearli": 7, "ident": 7, "zmesh_r": 7, "zmesh_im": 7, "everytim": 7, "devast": 7, "sinc": 7, "recompil": 7, "everi": 7, "strategi": 7, "push": [7, 9], "invok": 7, "downstream": 7, "under": 7, "hood": 7, "exist": 7, "pointer": 7, "those": 7, "therefor": 7, "being": 7, "sent": [7, 8], "coupl": 7, "block": 7, "gpu_escape_time_zn_plus_c": 7, "z_mesh_r": 7, "z_mesh_im": 7, "creal": 7, "cimag": 7, "tempreal": 7, "tempimag": 7, "znreal": 7, "znimag": 7, "confirm": 7, "16": [7, 8], "emphas": 7, "raw": [7, 8], "applic": [7, 8], "host": [7, 8], "lim": 7, "sole": 7, "c_re": 7, "c_im": 7, "comparison": 7, "dramat": [7, 9], "batch": 7, "host_var": 7, "display_t": 7, "ljust": 7, "rjust": 7, "nnote": 7, "shown": [7, 9], "quickli": 7, "viewer": 7, "realli": 7, "mpl": 7, "fractalexplor": 7, "__init__": 7, "self": 7, "clim": 7, "img_siz": 7, "is_press": 7, "connect": [7, 8, 9], "cidpress": 7, "canva": 7, "mpl_connect": 7, "button_press_ev": 7, "on_press": 7, "cidreleas": 7, "button_release_ev": 7, "on_releas": 7, "cidmot": 7, "motion_notify_ev": 7, "on_mot": 7, "event": 7, "xdata": 7, "ydata": 7, "set_text": 7, "set_data": 7, "disconnect": 7, "mpl_disconnect": 7, "800": 7, "fs": [7, 8], "common": 7, "mous": 7, "76": 7, "60": 7, "2540": 7, "1440": 7, "colormap_nam": 7, "hot": 7, "creation": 7, "non": 7, "mesh_wp_r": 7, "mesh_wp_im": 7, "chosen": 7, "troubl": 7, "eleg": 7, "txt_part": 7, "get_text": 7, "norm": 7, "vmin": 7, "vmax": 7, "get_cmap": 7, "save_fil": 7, "wallpaper_fractal_": 7, "readi": 7, "truncat": 7, "larger": 7, "intend": 7, "imsav": 7, "eyebal": 7, "score": 7, "correl": 7, "otherwis": 7, "hypothesi": 7, "frequenc": 7, "bad": 7, "fourier": 7, "densiti": 7, "edg": 7, "indic": 7, "componenet": 7, "beauti": 7, "solut": 7, "tini": 7, "amount": [7, 9], "kick": 7, "diagram": 7, "oversimplif": 7, "accur": 7, "portrai": 7, "fft_domain": 7, "fftshift": 7, "fft2": 7, "log10": 7, "titl": 7, "resuabl": 7, "score_mask": 7, "evaluate_fract": 7, "spectral": 7, "good_fract": 7, "bad_fract": 7, "good_log_fft": 7, "bad_log_fft": 7, "good_scor": 7, "bad_scor": 7, "4i": 7, "absout": 7, "term": 7, "lot": [7, 9], "But": [7, 9], "c_map": 7, "disp_good_c_map": 7, "turn": [7, 9], "qualiti": 7, "fratal": 7, "sourc": [7, 8], "lin_max": 7, "highest_freq_fract": 7, "max_iter_escap": 7, "gpu_escape_time_zn3plusc": 7, "escape_time_zn3minus1": 7, "equat": 7, "470": 7, "580i": 7, "560": 7, "230i": 7, "550": 7, "292i": 7, "535": 7, "378i": 7, "465": 7, "028i": 7, "56": 7, "23": 7, "23i": 7, "fs3": 7, "window": [7, 9], "skew": 7, "zmesh_wp_r": 7, "zmesh_wp_im": 7, "wallpaper_fract": 7, "wallpaper_fractal_zn3p1": 7, "z3_c_map": 7, "t_start": 7, "evalu": 7, "time_per_fract": 7, "4f": 7, "z3_disp_good_c_map": 7, "xp": 7, "depend": [7, 9], "num_test": 7, "test_fft_modul": 7, "random_input": 7, "n_test": 7, "get_array_modul": 7, "modnam": 7, "strip": [7, 8], "my_arrai": 7, "noqa": 8, "f401": 8, "mpi4pi": 8, "wget": 8, "fem": 8, "github": 8, "sh": 8, "tmp": 8, "bash": 8, "2023": 8, "08": 8, "28": 8, "02": 8, "38": 8, "resolv": 8, "185": 8, "199": 8, "108": 8, "153": 8, "109": 8, "110": 8, "443": 8, "request": 8, "await": 8, "respons": 8, "200": 8, "ok": 8, "length": 8, "2595": 8, "5k": 8, "kb": 8, "53k": 8, "001": 8, "33": 8, "mb": 8, "install_prefix": 8, "awk": 8, "nf": 8, "echo": 8, "install_prefix_depth": 8, "project_nam": 8, "share_prefix": 8, "mpi4py_instal": 8, "gcc_install_script_path": 8, "com": 8, "195f63f": 8, "gcc": 8, "gcc_install_script_download": 8, "82": 8, "114": 8, "302": 8, "githubusercont": 8, "195f63f4df27f26c318d8d7680f25e3cf2804418": 8, "133": 8, "111": 8, "7933": 8, "7k": 8, "plain": 8, "75k": 8, "0s": 8, "72": 8, "gcc_instal": 8, "lib64": 8, "rsync": 8, "avz": 8, "send": 8, "increment": 8, "libblosc2": 8, "pkgconfig": 8, "blosc2": 8, "pc": 8, "112": 8, "132": 8, "byte": 8, "receiv": 8, "242": 8, "224": 8, "748": 8, "00": 8, "408": 8, "834": 8, "66": 8, "rf": 8, "ln": 8, "gcc_archive_path": 8, "download": 8, "20230817": 8, "174604": 8, "ca3c9ea": 8, "tar": 8, "gz": 8, "gcc_archive_download": 8, "product": 8, "asset": 8, "2e65b": 8, "370599515": 8, "fca4b572": 8, "0b62": 8, "4028": 8, "9b13": 8, "c670920289a0": 8, "amz": 8, "aws4": 8, "hmac": 8, "sha256": 8, "credenti": 8, "akiaiwnjyax4csveh53a": 8, "2f20230828": 8, "2fu": 8, "east": 8, "2fs3": 8, "2faws4_request": 8, "date": 8, "20230828t023858z": 8, "expir": 8, "300": 8, "signatur": 8, "911dc59877cdc5c293ec31bc3425c61bc50a8384e2164b0dc6c875025e9691e2": 8, "signedhead": 8, "actor_id": 8, "key_id": 8, "repo_id": 8, "content": 8, "disposit": 8, "attach": 8, "3b": 8, "20filenam": 8, "3dgcc": 8, "2foctet": 8, "58": 8, "687604540": 8, "656m": 8, "octet": 8, "ta": 8, "655": 8, "75m": 8, "165mb": 8, "1s": 8, "39": 8, "162": 8, "xzf": 8, "directori": 8, "apt": 8, "qq": 8, "zlib1g": 8, "newest": 8, "dfsg": 8, "2ubuntu9": 8, "upgrad": 8, "newli": 8, "legacy_gpp": 8, "bin": 8, "dumpvers": 8, "legacy_gcc_vers": 8, "altern": 8, "provid": [8, 9], "nm": 8, "ranlib": 8, "x86_64": 8, "gnu": 8, "gcc_version": 8, "gfortran": 8, "libstdcxx_replac": 8, "python_exec": 8, "dirnam": 8, "python_exec_dir": 8, "objdump": 8, "sed": 8, "path": [8, 9], "origin": 8, "grep": 8, "python_rpath": 8, "install_prefix_rpath": 8, "basenam": 8, "libstdc": 8, "libstdcxx_system_vers": 8, "gdb": 8, "libstdcxx_install_prefix_vers": 8, "mkdir": 8, "touch": 8, "mpi4py_archive_path": 8, "182017": 8, "mpi4py_archive_download": 8, "064b1b63": 8, "d0ef": 8, "4947": 8, "b140": 8, "1ac16083a4b1": 8, "20230828t023920z": 8, "950c119e3b7e615832efc9fbeac3854337898c3db66d15c3b00d9e3b584bdfa7": 8, "3dmpi4pi": 8, "8848171": 8, "4m": 8, "44m": 8, "03": 8, "272": 8, "command": 8, "mpicc": 8, "mpi_lib": 8, "libmca": 8, "libmpi": 8, "libompi": 8, "libopen": 8, "pal": 8, "rte": 8, "ompi": 8, "libmca_common_monitor": 8, "50": 8, "libmca_common_ofi": 8, "libmca_common_ompio": 8, "29": 8, "libmca_common_sm": 8, "libmca_common_ucx": 8, "libmca_common_verb": 8, "libmca_common_dstor": 8, "libmpi_cxx": 8, "libmpi_java": 8, "libmpi_mpifh": 8, "libmpi_usempif08": 8, "libmpi_usempi_ignore_tkr": 8, "libompitrac": 8, "ompi_monitoring_prof": 8, "mpi": 8, "assert": 8, "comm_world": 8, "is_initi": 8, "is_fin": 8, "helloworld": 8, "cpp": 8, "argc": 8, "char": 8, "argv": 8, "environ": [8, 9], "mpi_init": 8, "null": 8, "world_siz": 8, "mpi_comm_s": 8, "mpi_comm_world": 8, "rank": 8, "world_rank": 8, "mpi_comm_rank": 8, "printf": 8, "hello": 8, "world": 8, "mpi_fin": 8, "mpicxx": 8, "static": 8, "mpirun": 8, "root": 8, "oversubscrib": 8, "growth": 9, "logic": 9, "slow": 9, "year": 9, "futur": 9, "inher": 9, "programm": 9, "respond": 9, "slowdown": 9, "seek": 9, "maker": 9, "embed": 9, "challeng": 9, "exploit": 9, "simultan": 9, "particularli": 9, "handl": 9, "lectur": 9, "discuss": 9, "tool": 9, "quantit": 9, "econom": 9, "textbook": 9, "written": 9, "keep": 9, "briefli": 9, "review": 9, "kind": 9, "commonli": 9, "pro": 9, "con": 9, "context": 9, "carri": 9, "latter": 9, "usual": 9, "With": 9, "struggl": 9, "low": 9, "lightweight": 9, "resourc": 9, "fact": 9, "extrem": 9, "conveni": 9, "hand": 9, "suffic": 9, "realiz": 9, "assum": 9, "latest": 9, "anaconda": 9, "action": 9, "next": 9, "piec": 9, "eigenvalu": 9, "maxim": 9, "previous": 9, "5000": 9, "mac": 9, "perfmon": 9, "observ": 9, "load": 9, "bump": 9, "At": 9, "least": 9, "successfulli": 9, "reason": 9, "f_vec": 9, "quit": 9, "avoid": 9, "entir": 9, "pair": 9, "gain": 9, "inform": 9, "plu": 9, "target": 9, "float64": 9, "significantli": 9, "outer": 9, "insid": 9, "independ": 9, "fail": 9, "problemat": 9, "inner": 9, "ordinari": 9, "prang": 9, "hold": 9, "earlier": 9, "effort": 9, "speak": 9, "rel": 9, "spread": 9, "suitabl": 9, "nontrivi": 9, "someth": 9, "substanti": 9, "100_000_000": 9, "njit": 9, "calculate_pi": 9, "1_000_000": 9, "u": 9, "sqrt": 9, "area_estim": 9, "divid": 9, "user": 9, "137": 9, "sy": 9, "92": 9, "141": 9, "14206": 9, "622": 9, "633": 9, "678": 9, "142072": 9, "annot": 9, "workstat": 9, "factor": 9, "mainli": 9}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"py": 0, "parallel": [0, 4, 5, 9], "basic": [0, 2], "vector": [0, 2, 5], "multiprocess": [0, 3, 5, 9], "multithread": [0, 5, 9], "gpu": [0, 1, 5, 6, 7], "tpu": 0, "let": 0, "make": 0, "runtim": [0, 1, 2, 3], "comparison": [0, 2], "plot": [0, 1, 3], "fractal": [1, 7], "gener": [1, 7], "cpu": [1, 7], "numpi": [1, 2, 5, 7, 9], "implement": [1, 2, 3, 7], "cupi": [1, 6, 7], "anim": 1, "question": [1, 2, 3], "n": 2, "bodi": 2, "simul": 2, "loop": 2, "\u03c0": 3, "estim": 3, "base": 3, "techniqu": 4, "python": 4, "overview": [5, 9], "flynn": 5, "s": 5, "taxonomi": 5, "type": [5, 9], "vs": 5, "implicit": [5, 9], "comput": 5, "numba": [6, 9], "pycuda": 6, "fun": 7, "With": 7, "background": 7, "hardwar": 7, "summari": 7, "find": 7, "aggreg": 7, "from": 7, "run": 7, "notebook": 7, "saniti": 7, "check": 7, "2": 7, "speed": 7, "test": 7, "3": 7, "defin": 7, "cuda": 7, "kernel": 7, "us": 7, "elementwisekernel": 7, "time": 7, "4": 7, "interact": 7, "creat": 7, "desktop": 7, "wallpap": 7, "your": 7, "5": 7, "search": 7, "simpl": 7, "data": 7, "pipelin": 7, "through": 7, "The": 7, "good": 7, "map": 7, "cubic": 7, "z": 7, "sup": 7, "c": 7, "variant": 7, "fft": 7, "agnost": 7, "code": 7, "content": 9, "advantag": 9, "disadvantag": 9, "A": 9, "matrix": 9, "oper": 9, "ufunc": 9, "warn": 9, "exercis": 9, "1": 9, "solut": 9}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})