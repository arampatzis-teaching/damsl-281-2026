"""Functions run by the pools in the notebooks of this folder.

They live in a module because a process pool starts a fresh interpreter for
every worker and imports the function by name. A function defined in a notebook
cell cannot be imported that way.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def cpu_work(n):
    """
    Run pure Python arithmetic, which holds the interpreter lock.

    Parameters
    ----------
    n : int
        Number of loop steps.

    Returns
    -------
    int
        Sum of (i * i) % 97 over i = 0, ..., n - 1.
    """
    s = 0
    for i in range(n):
        s += (i * i) % 97
    return s


def wait(seconds=0.2):
    """
    Sleep, standing for reading a file or waiting for an answer over the network.

    Parameters
    ----------
    seconds : float, optional
        Time to sleep, in seconds.

    Returns
    -------
    float
        The same `seconds`.
    """
    time.sleep(seconds)
    return seconds


def elementwise(a):
    """
    Do work that leaves Python. The compiled loop of np.sin releases the lock.

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------
    float
        Sum of the sines of the entries of `a`.
    """
    return float(np.sin(a).sum())


def harmonics_shared(job):
    """
    Compute `harmonics` of a piece read from a shared memory block.

    Parameters
    ----------
    job : tuple
        ``(name, shape, dtype_str, start, stop)``: the name of the shared
        memory block, the shape and dtype name of the array stored in it, and
        the first and one-past-last row of the slice.

    Returns
    -------
    float
        `harmonics` of the slice.
    """
    from multiprocessing import shared_memory

    name, shape, dtype_str, start, stop = job
    shm = shared_memory.SharedMemory(name=name)
    try:
        data = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        piece = data[start:stop]
        return harmonics(piece)
    finally:
        shm.close()


def harmonics(piece, terms=5):
    """
    Do real work per entry, so that the workers are not limited by the memory.

    Parameters
    ----------
    piece : ndarray
        Input array.
    terms : int, optional
        Number of harmonics.

    Returns
    -------
    float
        Sum over the entries x of sin(x) + sin(2x) + ... + sin(terms x).
    """
    total = np.zeros_like(piece)
    for k in range(1, terms + 1):
        total += np.sin(k * piece)
    return float(total.sum())


def harmonics_sent(piece):
    """
    Compute `harmonics` of a piece that was copied to the worker.

    Parameters
    ----------
    piece : ndarray
        Input array, copied to the worker.

    Returns
    -------
    float
        `harmonics` of `piece`.
    """
    return harmonics(piece)


def warm_pool(n_workers):
    """
    Start a process pool and make sure that all its workers are running.

    A pool starts its workers only when there is work for them. Giving each
    worker a job of 0.1 s forces the pool to start all of them, so that a
    timing that follows measures the work and not the start-up. The workers are
    started with spawn, as in the notebooks.

    Parameters
    ----------
    n_workers : int
        Number of worker processes.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        The pool, with all its workers started. Use it in a ``with`` block.
    """
    pool = ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
    )
    list(pool.map(wait, [0.1] * n_workers))
    return pool
