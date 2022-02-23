import numpy as np
import cv2

import multiprocessing as mp
from multiprocessing import connection
import queue
import time

import atexit

from enum import Enum

class DebuggerElement(Enum):
    StopProcess=0
    Image=1

class _CVLogChild:
    def __init__(self, conn: connection.Connection):
        self._conn = conn
        self._num_cv_images = 0
        self._shutdown = False

    def _process_input(self, e):
        kind, data = e
        if kind == DebuggerElement.StopProcess:
            self._shutdown = True
        elif kind == DebuggerElement.Image:
            img, name = data
            cv2.imshow(name, img)
            self._num_cv_images += 1

    def run (self):
        while not self._shutdown:
            if self._num_cv_images > 0:
                cv2.waitKey(100)
            
            if self._conn.poll(0.01):
                e = self._conn.recv()
                self._process_input (e)

class CVLog:
    def __init__(self):
        self.child = None

    def shutdown(self):
        self.parent_conn.send((DebuggerElement.StopProcess, None))

    def image(self, img: np.ndarray, name: str):
        self._ensure_started()
        self.parent_conn.send((DebuggerElement.Image, (img, name)))

    def _ensure_started (self):
        if not self.child:
            self._start_child()

    def _start_child (self):
        self.ctx = mp.get_context('fork')        
        self.parent_conn, child_conn = mp.Pipe()
        self.child = self.ctx.Process(target=CVLog._run_child, args=(child_conn,))
        self.child.start ()

        # Make sure that we'll kill the logger when shutting down the parent process.
        atexit.register(CVLog._cvlog_shutdown, self)

    def _cvlog_shutdown(this_cvlog):
        this_cvlog.shutdown()

    def _run_child(conn: connection.Connection):
        processor = _CVLogChild(conn)
        processor.run ()

cvlog = CVLog()
