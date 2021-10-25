import sys
import time
import traceback
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects


if __name__ == '__main__':
    #%%
    def func_async(i, *args):
        return 2 * i

    res = Parallel(n_jobs=2)(delayed(func_async)(21) for _ in range(1))
    print(res)

    #%%
    def func_async(i, *args):
        return 2 * i


    # We have to pass an extra argument with a large list (or another large python
    # object).
    large_list = list(range(1000000))

    t_start = time.time()
    Parallel(n_jobs=2)(delayed(func_async)(21, large_list) for _ in range(1))
    print("With loky backend and cloudpickle serialization: {:.3f}s".format(time.time() - t_start))

    #%%
    set_loky_pickler('pickle')
    t_start = time.time()
    Parallel(n_jobs=2)(delayed(id)(large_list) for _ in range(1))
    print("With pickle serialization: {:.3f}s".format(time.time() - t_start))

          #%%
    def func_async(i, *args):
        return 2 * i


    try:
        Parallel(n_jobs=2)(delayed(func_async)(21, large_list) for _ in range(1))
    except Exception:
        traceback.print_exc(file=sys.stdout)

        #%%


    @delayed
    @wrap_non_picklable_objects
    def func_async_wrapped(i, *args):
        return 2 * i


    t_start = time.time()
    Parallel(n_jobs=2)(func_async_wrapped(21, large_list) for _ in range(1))
    print("With pickle from stdlib and wrapper: {:.3f}s"
          .format(time.time() - t_start))

    #%%
    # Reset the loky_pickler to avoid border effects with other examples in
    # sphinx-gallery.
    set_loky_pickler()