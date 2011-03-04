import time
import functools

def timing(func):
  def wrapper(*args, **argd):
    start_time = time.time()
    ret = func(*args, **argd)
    print 'function: %s used %.2f seconds' % (func.__name__, time.time()-start_time)
    return ret
  functools.update_wrapper(wrapper, func)
  
  return wrapper

