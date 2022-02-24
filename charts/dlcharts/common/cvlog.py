import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl

import multiprocessing as mp
from multiprocessing import connection
import queue
import time
import pickle
import sys

import atexit

from enum import Enum

from icecream import ic
class DebuggerElement(Enum):
    StopProcess=0
    StopWhenAllWindowsClosed=1
    Image=2
    Figure=3

class _CVLogChild:
    def __init__(self, conn: connection.Connection):
        self._conn = conn
        self._num_cv_images = 0
        self._shutdown = False
        self._stop_when_all_windows_closed = False
        self._figures_by_name = dict()

    def _process_input(self, e):
        kind, data = e
        if kind == DebuggerElement.StopProcess:
            self._shutdown = True
        elif kind == DebuggerElement.StopWhenAllWindowsClosed:
            self._stop_when_all_windows_closed = True
        elif kind == DebuggerElement.Image:
            img, name = data
            cv2.imshow(name, img)
            self._num_cv_images += 1
        elif kind == DebuggerElement.Figure:
            fig, name = data
            if name in self._figures_by_name:
                plt.close (self._figures_by_name[name])
            self._figures_by_name[name] = fig
            fig.canvas.manager.set_window_title(name)
            fig.canvas.mpl_connect('close_event', lambda e: self._on_fig_close(name))
            fig.show ()

    def _on_fig_close(self, name):
        del self._figures_by_name[name]

    def _shouldStop (self):
        # not dict returns True if empty
        if self._num_cv_images < 1 and not self._figures_by_name and self._stop_when_all_windows_closed:
            return True
        return self._shutdown

    def run (self):        
        while not self._shouldStop():
            if self._num_cv_images > 0:
                k = cv2.waitKey(5)
                if k&0xff == ord('q'):
                    cv2.destroyAllWindows()
                    self._num_cv_images = 0

            if self._figures_by_name:
                # This would always bring the window to front, which is not what I want.
                # plt.pause(0.005)
                manager = plt.get_current_fig_manager()
                if manager is not None:
                    manager.canvas.figure.canvas.flush_events()
                    manager.canvas.figure.canvas.draw_idle()
            
            if self._conn.poll(0.005):
                e = self._conn.recv()
                self._process_input (e)            

class CVLog:
    def __init__(self):
        # Start the subprocess right away, to make sure that we won't fork
        # the program too late where there is already a lot of stuff in memory.
        # In particular matplotlib stuff can be problematic if we share the
        # memory of the main process.
        # But keep it disabled by default until explicitly enabled.
        self._enabled = False
        self._start_child ()

    @property
    def enabled(self): return self._enabled

    @enabled.setter
    def enabled(self, value): self._enabled = value

    def waitUntilWindowsAreClosed(self):
        self.parent_conn.send((DebuggerElement.StopWhenAllWindowsClosed, None))
        self.child.join()
        self.child = None

    def shutdown(self):
        if self.child:
            self.parent_conn.send((DebuggerElement.StopProcess, None))

    def image(self, img: np.ndarray, name: str = "CVLog Image"):
        if not self._enabled:
            return
        self.parent_conn.send((DebuggerElement.Image, (img, name)))

    def plot(self, fig: mpl.figure.Figure, name: str = "CVLog Plot"):
        """Show a matplotlib figure
        
        Sample code
            with plt.ioff():
                fig,ax = plt.subplots(1,1)
                ax.plot([1,2,3], [1,4,9])
                cvlog.plot(fig)
        """
        if not self._enabled:
            return
        self.parent_conn.send((DebuggerElement.Figure, (fig, name)))

    def _start_child (self):
        if plt.get_fignums():
            # If you create figures before forking, then the shared memory will
            # make a mess and freeze the subprocess.
            # Unfortunately this still does not catch whether a figure was created
            # and already closed, which is also a problem.
            raise Exception("You need to call start before creating any matplotlib figure.")

        self.ctx = mp.get_context()
        self.parent_conn, child_conn = mp.Pipe()
        self.child = self.ctx.Process(target=CVLog._run_child, args=(child_conn,))
        self.child.start ()

        # Make sure that we'll kill the logger when shutting down the parent process.
        atexit.register(CVLog._cvlog_shutdown, self)

    def _cvlog_shutdown(this_cvlog):
        this_cvlog.waitUntilWindowsAreClosed()

    def _run_child(conn: connection.Connection):
        processor = _CVLogChild(conn)
        processor.run ()

cvlog = CVLog()
