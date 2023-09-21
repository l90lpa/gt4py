
from functools import partial

from jax import config, jit, vjp
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
import gt4py.storage

####### GT4Py Settings

backend = "numpy"  # options: "numpy", "gt:cpu_ifirst", "gt:cpu_kfirst", "gt:gpu", "dace:cpu", "dace:gpu"
backend_opts = {"verbose": True} if backend.startswith("gt") else {}
dtype = np.float64
origin = (0, 0, 0)
rebuild = True

####### Compilation

@partial(jit, static_argnames=['validate_args', 'origin', 'domain'])
@gtscript.stencil(backend=backend, rebuild=rebuild, **backend_opts)
def localsum(
    field: gtscript.Field[dtype],
    result: gtscript.Field[dtype],
    *,
    weight: dtype = 2.0,
):
    with computation(PARALLEL), interval(...):
        result = weight * (field[1, 0, 0] + field[0, 1, 0] + field[-1, 0, 0] + field[0, -1, 0])

####### Set-up and initialize field

nx = ny = 4

field  = jnp.array(gt4py.storage.ones((nx, ny, 1), dtype, backend=backend, aligned_index=(1, 1, 0)))
result = jnp.array(gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=(1, 1, 0)))
weight = 2.0

####### Evaluate stencil

assert localsum._cache_size() == 0

field_new, result_new = localsum(field=field, result=result, weight=weight, validate_args=False, origin=(1, 1, 0), domain=(nx-2, ny-2, 1))

# Check first application results
assert np.array_equal(np.squeeze(field_new), np.array([[1, 1, 1, 1,],
                                                       [1, 1, 1, 1,],
                                                       [1, 1, 1, 1,],
                                                       [1, 1, 1, 1,],], dtype=dtype))
assert np.array_equal(np.squeeze(result_new), np.array([[0, 0, 0, 0,],
                                                        [0, 8, 8, 0,],
                                                        [0, 8, 8, 0,],
                                                        [0, 0, 0, 0,],], dtype=dtype))
assert localsum._cache_size() == 1

field_new, result_new = localsum(field=result_new, result=result_new, weight=weight, validate_args=False, origin=(1, 1, 0), domain=(nx-2, ny-2, 1))

# Check second application results
assert np.array_equal(np.squeeze(field_new), np.array([[0, 0, 0, 0,],
                                                       [0, 8, 8, 0,],
                                                       [0, 8, 8, 0,],
                                                       [0, 0, 0, 0,],], dtype=dtype))
assert np.array_equal(np.squeeze(result_new), np.array([[0,  0,  0,  0,],
                                                        [0, 32, 32,  0,],
                                                        [0, 32, 32,  0,],
                                                        [0,  0,  0,  0,],], dtype=dtype))
assert localsum._cache_size() == 1

####### Wrap `localsum` stencil to make positional args interface

def localsum_positional_args(field, result, weight):
    field_new, result_new = localsum(field=field, result=result, weight=weight, validate_args=False, origin=(1, 1, 0), domain=(nx-2, ny-2, 1))
    return field_new, result_new

####### Set-up and initialize field cotangent field

Dresult = jnp.array(gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=(1, 1, 0)))
Dresult = Dresult.at[1,1,0].set(1.0)
Dfield = jnp.array(gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=(1, 1, 0)))

####### Construct vjp operator and primals

primals, vjp_localsum = vjp(localsum_positional_args, field, result, weight)

####### Evaluate vjp operator

cotangents = vjp_localsum((Dfield, Dresult))

####### Check results

# Check output primals
assert np.array_equal(np.squeeze(primals[0]), np.array([[1, 1, 1, 1,],
                                                        [1, 1, 1, 1,],
                                                        [1, 1, 1, 1,],
                                                        [1, 1, 1, 1,],], dtype=dtype))
assert np.array_equal(np.squeeze(primals[1]), np.array([[0, 0, 0, 0,],
                                                        [0, 8, 8, 0,],
                                                        [0, 8, 8, 0,],
                                                        [0, 0, 0, 0,],], dtype=dtype))

# Check output cotangents
assert np.array_equal(np.squeeze(cotangents[0]), np.array([[0, 2, 0, 0,],
                                                           [2, 0, 2, 0,],
                                                           [0, 2, 0, 0,],
                                                           [0, 0, 0, 0,],], dtype=dtype))
assert np.array_equal(np.squeeze(cotangents[1]), np.array([[0, 0, 0, 0,],
                                                           [0, 0, 0, 0,],
                                                           [0, 0, 0, 0,],
                                                           [0, 0, 0, 0,],], dtype=dtype))
assert cotangents[2] == 4.0

        